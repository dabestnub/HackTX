function buildCoachMessages(payload = {}) {
  const target = String(payload.target || "").slice(0, 2).toUpperCase();
  const predicted = String(payload.predicted || "").slice(0, 2).toUpperCase();
  const errors = Array.isArray(payload.errors) ? payload.errors.slice(0, 4) : [];

  const system = [
    "You are a beginner ASL fingerspelling coach.",
    "Return exactly one concise sentence, at most 18 words.",
    "Include one encouraging phrase and one concrete correction.",
    "Use plain language and avoid safety, medical, legal, or policy statements.",
  ].join(" ");

  const user = JSON.stringify({
    mode: "tutorial_feedback",
    target,
    predicted,
    confidence: Number(payload.confidence || 0),
    holdMs: Number(payload.holdMs || 0),
    errors,
  });

  return [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
}

function buildDetailedCoachMessages(payload = {}) {
  const target = String(payload.target || "").slice(0, 2).toUpperCase();
  const predicted = String(payload.predicted || "").slice(0, 2).toUpperCase();
  const errors = Array.isArray(payload.errors) ? payload.errors.slice(0, 6) : [];

  const system = [
    "You are a friendly ASL fingerspelling tutorial coach for beginners.",
    "Return 3 to 5 short lines.",
    "Include: what likely went wrong, one correction, one micro-drill, and one confidence tip.",
    "Use plain language and avoid medical/legal/safety advice.",
  ].join(" ");

  const user = JSON.stringify({
    mode: "tutorial_feedback_detailed",
    target,
    predicted,
    confidence: Number(payload.confidence || 0),
    holdMs: Number(payload.holdMs || 0),
    errors,
  });

  return [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
}

function buildCoachChatMessages(payload = {}) {
  const history = Array.isArray(payload.messages) ? payload.messages.slice(-10) : [];
  const normalizedHistory = history
    .filter((entry) => entry && typeof entry === "object")
    .map((entry) => ({
      role: entry.role === "assistant" ? "assistant" : "user",
      content: String(entry.content || "").slice(0, 600),
    }))
    .filter((entry) => entry.content.trim());
  const context = payload.context && typeof payload.context === "object" ? payload.context : {};
  const system = [
    "You are an interactive ASL fingerspelling coach in a chat box.",
    "Be warm, practical, and specific.",
    "Answer in at most 8 short lines.",
    "When helpful, include numbered steps and a 30-second practice drill.",
    "Avoid medical/legal/safety advice.",
  ].join(" ");
  const contextMessage = {
    role: "user",
    content: JSON.stringify({
      mode: "tutorial_chat_context",
      target: String(context.target || "").slice(0, 2).toUpperCase(),
      predicted: String(context.predicted || "").slice(0, 2).toUpperCase(),
      confidence: Number(context.confidence || 0),
      holdMs: Number(context.holdMs || 0),
      errors: Array.isArray(context.errors) ? context.errors.slice(0, 6) : [],
    }),
  };
  const latestUser = {
    role: "user",
    content: String(payload.message || "").slice(0, 800),
  };

  return [
    { role: "system", content: system },
    contextMessage,
    ...normalizedHistory,
    latestUser,
  ];
}

function buildCoachPhotoMessages(payload = {}) {
  const context = payload.context && typeof payload.context === "object" ? payload.context : {};
  const imageDataUrl = String(payload.imageDataUrl || "").slice(0, 2_500_000);
  const system = [
    "You are an ASL fingerspelling coach reviewing one user-uploaded hand photo.",
    "Provide specific, actionable feedback in 4-7 short lines.",
    "Include one praise point, 2 concrete corrections, and one 20-second drill.",
    "Avoid medical/legal/safety advice.",
  ].join(" ");
  const contextMessage = {
    role: "user",
    content: JSON.stringify({
      mode: "tutorial_photo_context",
      target: String(context.target || "").slice(0, 2).toUpperCase(),
      predicted: String(context.predicted || "").slice(0, 2).toUpperCase(),
      confidence: Number(context.confidence || 0),
      holdMs: Number(context.holdMs || 0),
      errors: Array.isArray(context.errors) ? context.errors.slice(0, 6) : [],
    }),
  };
  const userMessage = {
    role: "user",
    content: String(payload.message || "Please review this hand photo and coach me.").slice(0, 800),
  };

  return [
    { role: "system", content: system },
    contextMessage,
    {
      role: "user",
      content: [
        { type: "text", text: userMessage.content },
        { type: "input_image", image_data: imageDataUrl },
      ],
    },
  ];
}

function buildLetterImagePrompt(letter) {
  const target = String(letter || "").slice(0, 1).toUpperCase();
  const handshapeSpecs = {
    A: "Closed fist; thumb rests along the side of the index finger, not crossing in front.",
    B: "Palm forward; four fingers straight and together upward; thumb folded across the palm.",
    C: "Curved fingers and thumb form a clear open C-shape with visible gap.",
    D: "Index finger points straight up; thumb and middle/ring/pinky touch in a round shape.",
    E: "Fingertips curl down toward thumb; thumb crosses lightly under fingertips.",
    F: "Thumb and index fingertip touch to make a small circle; other three fingers extend up.",
    G: "Index finger points sideways with thumb parallel; other fingers folded down.",
    H: "Index and middle fingers extend together sideways; ring and pinky folded; thumb tucked.",
    I: "Pinky extends up; other fingers folded; thumb rests across folded fingers.",
    J: "Same handshape as I, shown as a static frame with pinky extended (no motion trail).",
    K: "Index and middle fingers form a V upward; thumb touches middle finger base; ring and pinky folded.",
    L: "Index points up and thumb points sideways to form a clear L; other fingers folded.",
    M: "Thumb tucked under index, middle, and ring fingers; three finger knuckles visible over thumb.",
    N: "Thumb tucked under index and middle fingers ONLY; exactly two finger knuckles visible over thumb, ring and pinky folded. Not M.",
    O: "All fingertips touch thumb to form a rounded O shape.",
    P: "Like K but angled downward: index points down, middle diagonally down, thumb contacting middle.",
    Q: "Like G but angled downward: index and thumb point down together; other fingers folded.",
    R: "Index and middle fingers cross; ring and pinky folded; thumb supports folded fingers.",
    S: "Tight fist with thumb folded across the front of fingers.",
    T: "Fist with thumb tucked between index and middle fingers.",
    U: "Index and middle fingers straight up and together; ring and pinky folded; thumb tucked.",
    V: "Index and middle fingers straight up in a V; ring and pinky folded; thumb tucked.",
    W: "Index, middle, and ring fingers straight up spread as W; pinky folded; thumb tucked.",
    X: "Index finger bent like a hook; other fingers folded; thumb supports the fist.",
    Y: "Thumb and pinky extend out; index/middle/ring folded.",
    Z: "Index finger extended as if tracing Z; use a static frame with index extended and other fingers folded.",
  };
  const shapeSpec = handshapeSpecs[target] || "Accurate static ASL fingerspelling handshape for the target letter.";
  return [
    `Create an educational reference image for ASL fingerspelling letter ${target}.`,
    shapeSpec,
    "Show one clear right hand centered in frame on a plain neutral background.",
    "Realistic hand anatomy, high contrast lighting, wrist visible, and all fingertips clearly visible.",
    "Important: produce the canonical ASL handshape for this exact letter, not a nearby letter variant.",
    "No text, no labels, no watermark, no logos.",
  ].join(" ");
}

function buildLetterImageRetryPrompt(letter, mismatchLetter = "") {
  const target = String(letter || "").slice(0, 1).toUpperCase();
  const mismatch = String(mismatchLetter || "").slice(0, 1).toUpperCase();
  const mismatchNote = /^[A-Z]$/.test(mismatch)
    ? `Previous attempt looked like ${mismatch}; correct this and output ${target} exactly.`
    : `Previous attempt was incorrect; output ${target} exactly.`;
  return [
    buildLetterImagePrompt(target),
    mismatchNote,
    `Critical distinction: this must be canonical ASL ${target}, not any neighboring handshape.`,
    "Center the hand and make finger/knuckle/thumb placement unambiguous.",
  ].join(" ");
}

function buildLetterImageVerificationMessages(letter, imageDataUrl) {
  const target = String(letter || "").slice(0, 1).toUpperCase();
  return [
    {
      role: "system",
      content: [
        "You are verifying a static ASL fingerspelling image.",
        "Return ONLY one uppercase letter A-Z if clearly identifiable.",
        "If not clear, return UNKNOWN.",
      ].join(" "),
    },
    {
      role: "user",
      content: [
        { type: "text", text: `Identify the ASL fingerspelling letter shown. Target expected: ${target}.` },
        { type: "input_image", image_data: String(imageDataUrl || "").slice(0, 2_500_000) },
      ],
    },
  ];
}

function buildLetterImageTargetCheckMessages(letter, imageDataUrl) {
  const target = String(letter || "").slice(0, 1).toUpperCase();
  return [
    {
      role: "system",
      content: [
        "You are validating whether an ASL fingerspelling image matches one target letter.",
        "Return ONLY YES or NO.",
        "Use NO if uncertain or ambiguous.",
      ].join(" "),
    },
    {
      role: "user",
      content: [
        { type: "text", text: `Does this image show canonical ASL fingerspelling letter ${target}?` },
        { type: "input_image", image_data: String(imageDataUrl || "").slice(0, 2_500_000) },
      ],
    },
  ];
}

function sanitizeAdvice(text, fallback) {
  if (!text || typeof text !== "string") return fallback;
  const trimmed = text.replace(/\s+/g, " ").trim();
  if (!trimmed) return fallback;
  return trimmed.split(" ").slice(0, 18).join(" ");
}

function sanitizeDetailedAdvice(text, fallback) {
  if (!text || typeof text !== "string") return fallback;
  const trimmed = text.replaceAll("\r", "").trim();
  if (!trimmed) return fallback;
  return trimmed.slice(0, 1200);
}

module.exports = {
  buildCoachMessages,
  buildDetailedCoachMessages,
  buildCoachChatMessages,
  buildCoachPhotoMessages,
  buildLetterImagePrompt,
  buildLetterImageRetryPrompt,
  buildLetterImageVerificationMessages,
  buildLetterImageTargetCheckMessages,
  sanitizeAdvice,
  sanitizeDetailedAdvice,
};
