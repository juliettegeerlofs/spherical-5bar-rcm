// bench_arduino.ino -- drives the two Iq1/Iq2 steppers (TMC2209, direct
// drive, 1.8 deg/full step, 1/64 microstepping) and reports the BNO055's
// fused orientation (quaternion) back over serial, so bench_run.py (on the
// PC) can drive the bench automatically for every pose it plans.
//
// Uses the Adafruit BNO055 (9-axis, onboard sensor fusion) -- replaces
// the earlier GY-521/MPU6050 version. The BNO055 fuses orientation
// continuously on its OWN onboard processor, independent of when the
// Arduino polls it -- so unlike the MPU6050_light version, there is no
// need to interleave IMU reads with the stepping loop at all. A single
// read after each move is both safe (no I2C calls during pulse
// generation) AND accurate (the chip was tracking the whole time anyway).
//
// PIN ASSIGNMENT -- CHECK/ADJUST against your actual wiring before use:
//   motor 1 (Iq1): STEP1, DIR1, EN1
//   motor 2 (Iq2): STEP2, DIR2, EN2
//   BNO055: SDA->A4, SCL->A5 (Nano hardware I2C, fixed), VCC->Vin
//
// Serial protocol (9600 baud), one command per line:
//   F                       disable both drivers so the mechanism can be
//                           turned by hand with no holding torque to
//                           fight -- send this before asking the user to
//                           position it at home. Replies "FREE".
//   Z                       re-enable the drivers (locking in whatever
//                           pose the hand-positioning left it at) and
//                           zero both axes there -- only send this while
//                           the mechanism is physically at Iq1=Iq2=0
//                           (home). Replies "ZEROED" then "ZQ,<w>,<x>,<y>,<z>"
//                           with the home quaternion.
//   G,<Iq1_deg>,<Iq2_deg>   move to that absolute pose (relative to the
//                           last Z), settle, reply with "<w>,<x>,<y>,<z>\n"
//                           (current orientation quaternion)
//
// Raw quaternions, not Euler yaw/pitch, go over the wire on purpose: the
// model side (bench_run.py's predict_angles()) computes theta_h/theta_v by
// rotating a fixed tool-axis vector through a full relative rotation and
// projecting it -- not by subtracting Euler angle components. Euler-angle
// subtraction (old version of this file) is only valid for a pure
// single-axis tip; this mechanism moves both joints together, so that
// composition-order bug alone was enough to produce tens of degrees of
// apparent "model error", made worse near the BNO055 Euler convention's
// roll=+-90 singularity. Sending quaternions and doing the matching
// rotate-and-project math in Python (see bench_run.py) keeps bench and
// model directly comparable.

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

// ---- pin assignment: CHECK AGAINST YOUR OWN WIRING ----
// Swapped (back) to compensate the physical wire-bundle swap -- confirmed
// necessary: with the ORIGINAL numbers, G,0,10 (iq1_deg=0, iq2_deg=10)
// moved the physical Iq1 joint instead of Iq2, i.e. the two are
// genuinely swapped at the connectors and this pin swap is the correct
// fix for THAT. (The bench_roundtrip2 motor-heat incident was a separate
// issue -- a stall against the motor mount near Iq2=-30 deg, most likely
// because the --iq2-floor table is stale for the new link 5 geometry;
// that needs --calibrate-iq2-floor, not a pin change.)
const int STEP1 = 5, DIR1 = 4, EN1 = 7;   // motor for Iq1
const int STEP2 = 2, DIR2 = 3, EN2 = 6;   // motor for Iq2

// ---- mechanics ----
const float FULL_STEPS_PER_REV = 200.0;          // 1.8 deg/step NEMA-11
const float MICROSTEPS = 64.0;                    // TMC2209, MS1=low MS2=high
const float STEPS_PER_DEG = (FULL_STEPS_PER_REV * MICROSTEPS) / 360.0;
const unsigned int STEP_PULSE_US = 30000;   // cruise-speed pulse width, was
                                            // 5000 then 15000 then 30000 --
                                            // each bump was a flat-speed
                                            // workaround for the real issue
                                            // below (RAMP_* finally
                                            // addresses that directly), kept
                                            // as the steady-state speed once
                                            // up to speed
const unsigned int RAMP_START_PULSE_US = 90000;   // first-step pulse width:
                                            // was 60000 (2x cruise) -- that
                                            // fixed the stall outright in
                                            // one repeat but it came back,
                                            // milder, in the next (+0.1 deg
                                            // offset at the worst jump vs
                                            // +3.4 deg) -- a stochastic
                                            // failure, not eliminated by a
                                            // 2x-slower cold start alone.
                                            // 3x slower here for more
                                            // margin; combined with the
                                            // smaller MAX_STEP_RAD in
                                            // bench_run.py (more of each
                                            // substep now falls inside the
                                            // ramp zone). Still stalled
                                            // occasionally even at flat
                                            // 30000us/step because EVERY
                                            // move -- big or small --
                                            // jumped straight from a dead
                                            // stop to full cruise speed
                                            // with no ramp. A NEMA-11's
                                            // pull-in torque (what it can
                                            // deliver starting from rest at
                                            // a given speed) is much lower
                                            // than its running torque at
                                            // that same speed, so a cold
                                            // start right at cruise speed is
                                            // exactly when sync is most
                                            // likely to be lost, with no
                                            // feedback to catch it.
const long RAMP_STEPS = 100;               // microsteps over which to ramp
                                            // from RAMP_START_PULSE_US down
                                            // to STEP_PULSE_US (and back up
                                            // at the far end) -- was 60
                                            // (~1.7 deg), now ~2.8 deg;
                                            // moves shorter than
                                            // 2*RAMP_STEPS get a triangular
                                            // profile (never reach full
                                            // cruise speed) instead of
                                            // being clipped
const unsigned long SETTLE_MS = 2000;   // was 300 -- give mechanical ringing
                                         // from the move time to damp out
                                         // before trusting the BNO055 reading
// DITHER_* removed -- rapid direction reversals (8-9 per pose) exposed a
// DIR-to-STEP setup-time issue on the TMC2209 (no delay was inserted
// between flipping DIR and the first STEP pulse, which the original code
// never exercised since it only changes direction once per pose) and
// caused Iq2 specifically to drift off its commanded position. Reverting
// until that's fixed properly.

long pos1 = 0, pos2 = 0;   // current position in microsteps, since the last Z

void moveTo(long target1, long target2) {
  long d1 = target1 - pos1, d2 = target2 - pos2;
  digitalWrite(DIR1, d1 >= 0 ? HIGH : LOW);
  digitalWrite(DIR2, d2 >= 0 ? HIGH : LOW);
  long n1 = abs(d1), n2 = abs(d2);
  long n = max(n1, n2);
  if (n == 0) { pos1 = target1; pos2 = target2; return; }
  // symmetric ramp zone at each end of the move, clipped to n/2 so a short
  // move ramps up and immediately back down (triangular) instead of
  // overshooting into a flat cruise section it never reaches
  long ramp = min(RAMP_STEPS, n / 2);
  // both motors reach their target together: a simple DDA so the slower
  // axis gets its steps spread evenly across the move instead of all at
  // the start or end
  long acc1 = 0, acc2 = 0;
  for (long i = 0; i < n; i++) {
    acc1 += n1; acc2 += n2;
    bool p1 = acc1 >= n; if (p1) acc1 -= n;
    bool p2 = acc2 >= n; if (p2) acc2 -= n;
    if (p1) digitalWrite(STEP1, HIGH);
    if (p2) digitalWrite(STEP2, HIGH);
    delayMicroseconds(3);
    if (p1) digitalWrite(STEP1, LOW);
    if (p2) digitalWrite(STEP2, LOW);
    // trapezoidal speed ramp -- slow at the start and end of EVERY move,
    // cruise in the middle, instead of a dead-stop jump straight to full
    // speed (see RAMP_START_PULSE_US above for why)
    long dist_to_edge = min(i, n - 1 - i);
    unsigned int pulse_us = STEP_PULSE_US;
    if (ramp > 0 && dist_to_edge < ramp) {
      pulse_us = RAMP_START_PULSE_US -
                 (RAMP_START_PULSE_US - STEP_PULSE_US) * dist_to_edge / ramp;
    }
    delayMicroseconds(pulse_us);
    // no IMU reads needed in here -- see the file header note
  }
  pos1 = target1; pos2 = target2;
}

void printQuat(const imu::Quaternion &q) {
  Serial.print(q.w(), 6); Serial.print(",");
  Serial.print(q.x(), 6); Serial.print(",");
  Serial.print(q.y(), 6); Serial.print(",");
  Serial.println(q.z(), 6);
}

void doFree() {
  // back to BOTH axes -- leaving Iq2 enabled (only Iq1 free) assumed Iq2
  // would reliably already be sitting near zero, but it isn't guaranteed:
  // it's whatever the PREVIOUS session's last move left it at, and with
  // no way to hand-correct it, "Z" was silently zeroing against a wrong
  // Iq2 reference. That's a worse problem (a corrupted zero for the
  // whole session) than the inconvenience it was solving (needing a free
  // hand for Enter), so both axes get freed again. Explicit/repeatable
  // version of what setup() does once at boot -- relying on setup()
  // alone meant this only ever worked the very first time, because
  // opening the serial port from Python doesn't reliably reset the board
  // on every run (no reset = setup() never re-runs = the drivers stay in
  // whatever state doZero()/doGoto() last left them: enabled, fighting a
  // hand trying to re-zero). Python now sends "F" right before every
  // "put it at home by hand" prompt instead of assuming a fresh boot.
  digitalWrite(EN1, HIGH);
  digitalWrite(EN2, HIGH);
  Serial.println("FREE");
}

void doZero() {
  // enabling only now, right as this exact hand-set pose becomes the
  // reference, means the manual zero is no longer snapped to the nearest
  // microstep the motor happened to be holding beforehand (see doFree()).
  digitalWrite(EN1, LOW);
  digitalWrite(EN2, LOW);
  pos1 = 0; pos2 = 0;
  delay(SETTLE_MS);
  imu::Quaternion q = bno.getQuat();
  Serial.println("ZEROED");
  Serial.print("ZQ,");
  printQuat(q);
}

void doGoto(float iq1_deg, float iq2_deg) {
  long t1 = lround(iq1_deg * STEPS_PER_DEG);
  long t2 = lround(iq2_deg * STEPS_PER_DEG);
  moveTo(t1, t2);
  delay(SETTLE_MS);   // let it settle mechanically
  imu::Quaternion q = bno.getQuat();
  printQuat(q);
}

void setup() {
  // dropped to 9600 (from 115200) as a test against the intermittent
  // single-character serial corruption seen throughout recent sessions --
  // each bit now takes ~12x longer, giving a short EMI glitch from the
  // steppers far less chance of landing on the exact sampling instant.
  // bench_run.py's --baud must match this.
  Serial.begin(9600);
  pinMode(STEP1, OUTPUT); pinMode(DIR1, OUTPUT); pinMode(EN1, OUTPUT);
  pinMode(STEP2, OUTPUT); pinMode(DIR2, OUTPUT); pinMode(EN2, OUTPUT);
  // TMC2209: LOW = driver enabled, HIGH = disabled (free to turn by hand).
  // Stay DISABLED until doZero() -- see its comment: enabling at boot
  // would mean the very first manual "put it at home" is fought by the
  // motor's own holding torque and springs back to its nearest microstep,
  // which is exactly the step-quantised zero this is meant to avoid.
  digitalWrite(EN1, HIGH);
  digitalWrite(EN2, HIGH);

  Serial.println("BOOT: pins done");

  // Trying NDOF again (magnetometer on) -- the earlier NDOF sessions showed
  // a growing, Iq2-specific drift (+0.48 -> +7.81 -> +11.39 deg across
  // sessions) blamed on magnetometer disturbance from the nearby stepper
  // motors; IMUPLUS (magnetometer off) made that go away. Since then the
  // motor wiring has been changed to twisted pairs, which reduces radiated
  // EMI -- retesting whether that was enough to make NDOF usable again. If
  // Iq2 drift is back/worse than the IMUPLUS round-trip baseline, revert to
  // OPERATION_MODE_IMUPLUS.
  if (!bno.begin(OPERATION_MODE_NDOF)) {
    Serial.println("BOOT: BNO055 NOT FOUND -- check wiring");
  } else {
    Serial.println("BOOT: bno.begin() done");
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  Serial.println("READY");
}

String line;

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      line.trim();
      if (line == "Z") {
        doZero();
      } else if (line == "F") {
        doFree();
      } else if (line.startsWith("G,")) {
        int comma = line.indexOf(',', 2);
        float a = line.substring(2, comma).toFloat();
        float b = line.substring(comma + 1).toFloat();
        doGoto(a, b);
      } else if (line.length() > 0) {
        // always answer something -- an unrecognised line used to be
        // silently dropped, which looked exactly like a frozen Arduino
        Serial.print("ERR:");
        Serial.println(line);
      }
      line = "";
    } else {
      line += c;
    }
  }
}
