const FALLBACKS = {
  no_hand: "Great start—place one hand fully in frame and hold the shape steady.",
  confidence_low: "Nice effort—hold your hand steadier and closer so the sign is clearer.",
  hold_too_short: "Good try—keep that handshape steady a little longer before switching.",
  index_too_straight: "Almost there—curl your index finger more to match the target shape.",
  fingers_not_together: "Good attempt—bring your fingers closer together and hold that shape.",
  thumb_too_far: "Nice try—move your thumb closer toward your index finger.",
  palm_too_rotated: "Great effort—rotate your palm slightly toward the camera and hold.",
};

function fallbackAdvice(errors = []) {
  const first = Array.isArray(errors) ? errors.find((error) => FALLBACKS[error]) : null;
  return first ? FALLBACKS[first] : "Nice try—hold your hand steady and match the target shape.";
}

module.exports = { fallbackAdvice, FALLBACKS };
