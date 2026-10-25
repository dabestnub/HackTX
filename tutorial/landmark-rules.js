const ERROR_HINTS = {
  no_hand: "Place one hand fully in frame.",
  confidence_low: "Hold your hand steadier and closer to the camera.",
  hold_too_short: "Keep the shape steady a bit longer.",
  index_too_straight: "Curl your index finger more.",
  fingers_not_together: "Bring your fingers closer together.",
  thumb_too_far: "Move your thumb closer to your index finger.",
  palm_too_rotated: "Rotate your palm a little toward the camera.",
};

function distance(a, b) {
  if (!a || !b) return Infinity;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function evaluateLandmarks({ target, predicted, confidence, holdMs, landmarks }) {
  const errors = [];
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    errors.push("no_hand");
    return { pass: false, score: 0, errors, primaryHint: ERROR_HINTS.no_hand };
  }

  const conf = Number.isFinite(confidence) ? confidence : 0;
  if (conf < 0.6) errors.push("confidence_low");
  if ((holdMs || 0) < 1200) errors.push("hold_too_short");

  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const wrist = landmarks[0];

  if (distance(indexTip, indexPip) > 0.08) errors.push("index_too_straight");
  if (distance(indexTip, middleTip) > 0.07) errors.push("fingers_not_together");
  if (distance(thumbTip, thumbIp) > 0.06) errors.push("thumb_too_far");
  if (Math.abs((wrist?.x || 0.5) - 0.5) > 0.35) errors.push("palm_too_rotated");

  const predictionMatches = typeof target === "string" && predicted === target;
  const pass = predictionMatches && errors.length <= 1;
  const penalty = Math.min(0.8, errors.length * 0.15);
  const score = Math.max(0, Math.min(1, conf - penalty + (predictionMatches ? 0.15 : -0.15)));
  const primary = errors[0] || null;

  return {
    pass,
    score,
    errors,
    primaryHint: primary ? (ERROR_HINTS[primary] || "Adjust your hand and try again.") : "Great job!",
  };
}

module.exports = { evaluateLandmarks, ERROR_HINTS };
