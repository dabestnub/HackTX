(function tutorialCoachModule() {
  function createTutorialCoach(context = {}) {
    let tutorialOverlay = null;
    const tutorialAdviceCache = new Map();
    const tutorialState = {
      minAdviceIntervalMs: 2500,
      lastAdviceAt: 0,
      failureByLetter: new Map(),
      chatHistory: [],
      chatBusy: false,
      lastSignal: null,
      pendingPhoto: null,
      minimized: false,
    };

    const tutorialChatMaxMessages = 24;

    function softmaxTopClass(logits, labels) {
      if (!logits || !labels?.length) return { label: "", score: 0 };
      let maxLogit = -Infinity;
      let predictedIndex = 0;
      for (let index = 0; index < logits.length; index += 1) {
        if (logits[index] > maxLogit) {
          maxLogit = logits[index];
          predictedIndex = index;
        }
      }
      let exponentialTotal = 0;
      for (const logit of logits) {
        exponentialTotal += Math.exp(logit - maxLogit);
      }
      const score = 1 / exponentialTotal;
      return { label: labels[predictedIndex] || "", score };
    }

    async function verifyGeneratedImageWithLocalTracking({ imageDataUrl, expectedLetter }) {
      const expected = String(expectedLetter || "").slice(0, 1).toUpperCase();
      if (!expected || !imageDataUrl) {
        return { passed: false, predicted: null, confidence: 0, reason: "missing-input" };
      }
      try {
        await context.loadRecognition();
        const imageLandmarker = context.getImageLandmarker?.();
        const classifierSession = context.getClassifierSession?.();
        const classLabels = context.getClassLabels?.();
        if (!imageLandmarker || !classifierSession || !classLabels) {
          return { passed: false, predicted: null, confidence: 0, reason: "recognition-unavailable" };
        }

        const image = await new Promise((resolve, reject) => {
          const element = new Image();
          element.onload = () => resolve(element);
          element.onerror = () => reject(new Error("image-load-failed"));
          element.src = imageDataUrl;
        });

        if (!image.width || !image.height) {
          return { passed: false, predicted: null, confidence: 0, reason: "invalid-image" };
        }

        const inferenceCanvas = context.inferenceCanvas;
        const inferenceContext = context.inferenceContext;
        inferenceCanvas.width = image.width;
        inferenceCanvas.height = image.height;
        inferenceContext.drawImage(image, 0, 0, inferenceCanvas.width, inferenceCanvas.height);

        const result = imageLandmarker.detect(inferenceCanvas);
        if (!result?.landmarks?.length) {
          return { passed: false, predicted: null, confidence: 0, reason: "no-hand" };
        }

        const input = new window.ort.Tensor("float32", context.normalizeLandmarks(result.landmarks[0]), [1, 63]);
        const output = await classifierSession.run({ landmarks: input });
        const top = softmaxTopClass(output?.logits?.data, classLabels);
        const passed = top.label === expected && top.score >= 0.55;
        return {
          passed,
          predicted: top.label || null,
          confidence: top.score || 0,
          reason: passed ? "ok" : "mismatch",
        };
      } catch {
        return { passed: false, predicted: null, confidence: 0, reason: "verification-error" };
      }
    }

    async function fetchLocallyVerifiedLetterImage(letter, maxAttempts = 2) {
      const cleanedLetter = String(letter || "").trim().slice(0, 1).toUpperCase();
      if (!/^[A-Z]$/.test(cleanedLetter)) {
        return { error: "Please provide one letter from A to Z." };
      }
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        const response = await fetch("/api/coach/image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ letter: cleanedLetter }),
        });
        if (!response.ok) {
          return { error: "I couldn't generate that image right now. Try again in a moment." };
        }
        const body = await response.json();
        const imageDataUrl = body?.image?.imageDataUrl;
        if (!imageDataUrl) continue;

        const localCheck = await verifyGeneratedImageWithLocalTracking({
          imageDataUrl,
          expectedLetter: cleanedLetter,
        });

        if (localCheck.passed) {
          return { imageDataUrl, letter: cleanedLetter, confidence: localCheck.confidence, attempts: attempt };
        }

        return {
          imageDataUrl,
          letter: cleanedLetter,
          confidence: localCheck.confidence || 0,
          attempts: attempt,
          warning: localCheck.predicted
            ? `Local check read ${localCheck.predicted}; showing best generated match.`
            : "Local check was uncertain; showing best generated match.",
        };
      }
      return { error: "I couldn't generate that image right now. Try again in a moment." };
    }

    function pushTutorialChat(role, content) {
      const text = String(content || "").trim();
      if (!text) return;
      tutorialState.chatHistory.push({ role, content: text });
      if (tutorialState.chatHistory.length > tutorialChatMaxMessages) {
        tutorialState.chatHistory.splice(0, tutorialState.chatHistory.length - tutorialChatMaxMessages);
      }
      const overlay = ensureTutorialOverlay();
      const line = document.createElement("p");
      line.style.cssText = role === "assistant"
        ? "margin:6px 0;padding:6px 8px;border-radius:8px;background:#1c2f48;color:#dff"
        : "margin:6px 0;padding:6px 8px;border-radius:8px;background:#2f2f2f;color:#fff;text-align:right";
      line.textContent = text;
      overlay.chatLog.append(line);
      overlay.chatLog.scrollTop = overlay.chatLog.scrollHeight;
    }

    function pushTutorialImage({ imageDataUrl, caption = "Coach reference" } = {}) {
      if (!imageDataUrl) return;
      const overlay = ensureTutorialOverlay();
      const figure = document.createElement("figure");
      figure.style.cssText = "margin:6px 0;padding:6px;border-radius:8px;background:#1c2f48";
      const image = document.createElement("img");
      image.src = imageDataUrl;
      image.alt = caption;
      image.style.cssText = "display:block;width:100%;max-width:220px;border-radius:8px";
      const label = document.createElement("figcaption");
      label.textContent = caption;
      label.style.cssText = "margin-top:4px;color:#dff;font-size:12px";
      figure.append(image, label);
      overlay.chatLog.append(figure);
      overlay.chatLog.scrollTop = overlay.chatLog.scrollHeight;
    }

    function setTutorialChatBusy(busy) {
      tutorialState.chatBusy = busy;
      const overlay = ensureTutorialOverlay();
      overlay.sendButton.disabled = busy;
      overlay.input.disabled = busy;
      overlay.sendButton.textContent = busy ? "Sending…" : "Send";
    }

    function setTutorialMinimized(minimized) {
      const overlay = ensureTutorialOverlay();
      tutorialState.minimized = Boolean(minimized);
      if (tutorialState.minimized) {
        overlay.header.hidden = true;
        overlay.content.hidden = true;
        overlay.pillButton.hidden = false;
        overlay.root.style.padding = "0";
        overlay.root.style.maxWidth = "none";
        overlay.root.style.borderRadius = "12px 0 0 12px";
        overlay.root.style.right = "0";
        overlay.root.style.bottom = "18px";
        overlay.root.style.background = "transparent";
        overlay.root.style.boxShadow = "none";
      } else {
        overlay.header.hidden = false;
        overlay.content.hidden = false;
        overlay.pillButton.hidden = true;
        overlay.root.style.padding = "12px";
        overlay.root.style.maxWidth = "320px";
        overlay.root.style.borderRadius = "10px";
        overlay.root.style.right = "12px";
        overlay.root.style.bottom = "12px";
        overlay.root.style.background = "#111";
        overlay.root.style.boxShadow = "0 8px 24px rgba(0,0,0,.35)";
      }
      overlay.minimizeButton.textContent = tutorialState.minimized ? "Open" : "Minimize";
    }

    function toggleTutorialMinimized() {
      setTutorialMinimized(!tutorialState.minimized);
    }

    function toggleTutorialChat(forceOpen) {
      const overlay = ensureTutorialOverlay();
      if (overlay.content.hidden) setTutorialMinimized(false);
      const open = typeof forceOpen === "boolean" ? forceOpen : overlay.chatPanel.hidden;
      overlay.chatPanel.hidden = !open;
      overlay.toggleButton.textContent = open ? "Hide chat" : "Open chat";
      if (open) overlay.input.focus();
    }

    async function requestTutorialLetterImage(letter) {
      const cleanedLetter = String(letter || "").trim().slice(0, 1).toUpperCase();
      if (!/^[A-Z]$/.test(cleanedLetter) || tutorialState.chatBusy) {
        pushTutorialChat("assistant", "Please provide one letter from A to Z.");
        return;
      }
      setTutorialChatBusy(true);
      try {
        const result = await fetchLocallyVerifiedLetterImage(cleanedLetter, 2);
        if (result.error) {
          pushTutorialChat("assistant", result.error);
          return;
        }
        if (result.warning) {
          pushTutorialChat("assistant", result.warning);
        }
        pushTutorialChat("assistant", `Here’s a reference image for letter ${cleanedLetter}.`);
        pushTutorialImage({ imageDataUrl: result.imageDataUrl, caption: `ASL reference for ${cleanedLetter}` });
      } catch {
        pushTutorialChat("assistant", "Image generation is currently unavailable.");
      } finally {
        setTutorialChatBusy(false);
      }
    }

    async function sendTutorialChatMessage(text = null, options = {}) {
      const overlay = ensureTutorialOverlay();
      const message = String(text ?? overlay.input.value).trim();
      const photo = tutorialState.pendingPhoto?.dataUrl || null;
      if ((!message && !photo) || tutorialState.chatBusy) return;
      overlay.input.value = "";
      toggleTutorialChat(true);
      if (message) pushTutorialChat("user", message);
      if (photo) {
        const uploadedName = tutorialState.pendingPhoto?.name;
        const uploadCaption = uploadedName ? `Uploaded photo · ${uploadedName}` : "Uploaded photo";
        pushTutorialImage({ imageDataUrl: photo, caption: uploadCaption });
      }
      setTutorialChatBusy(true);
      try {
        const response = await fetch("/api/coach/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode: "chat",
            message: message || (options.autoPhoto ? "Please review this photo." : ""),
            messages: tutorialState.chatHistory.slice(-10),
            imageDataUrl: photo,
            context: {
              target: tutorialState.lastSignal?.target || "",
              predicted: tutorialState.lastSignal?.predicted || "",
              confidence: tutorialState.lastSignal?.confidence || 0,
              holdMs: tutorialState.lastSignal?.holdMs || 0,
              errors: tutorialState.lastSignal?.errors || [],
            },
          }),
        });
        if (!response.ok) {
          pushTutorialChat("assistant", "I couldn’t reach the coach service right now. Try again in a moment.");
          return;
        }
        const body = await response.json();
        const reply = String(body?.reply || "").trim();
        pushTutorialChat("assistant", reply || "I’m here—tell me which letter feels hardest right now.");
        const imageLetter = String(body?.image?.letter || "").slice(0, 1).toUpperCase();
        if (body?.image?.imageDataUrl && /^[A-Z]$/.test(imageLetter)) {
          pushTutorialImage({ imageDataUrl: body.image.imageDataUrl, caption: `ASL reference for ${imageLetter}` });
        }
      } catch {
        pushTutorialChat("assistant", "Coach connection dropped. I can still help with local tips while you keep practicing.");
      } finally {
        tutorialState.pendingPhoto = null;
        overlay.attachmentStatus.textContent = "No photo attached";
        overlay.attachInput.value = "";
        setTutorialChatBusy(false);
      }
    }

    function tutorialDistance(a, b) {
      if (!a || !b) return Infinity;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = (a.z || 0) - (b.z || 0);
      return Math.hypot(dx, dy, dz);
    }

    function tutorialFallback(errors = []) {
      const hints = {
        no_hand: "Place one hand fully in frame.",
        confidence_low: "Hold your hand steadier and closer to the camera.",
        hold_too_short: "Keep the shape steady a little longer.",
        index_too_straight: "Curl your index finger more.",
        fingers_not_together: "Bring your fingers closer together.",
        thumb_too_far: "Move your thumb closer to your index finger.",
        palm_too_rotated: "Rotate your palm a little toward the camera.",
      };
      const first = errors.find((code) => hints[code]);
      return first ? hints[first] : "Nice try. Keep your hand steady and match the target shape.";
    }

    function evaluateTutorialSignals({ target, predicted, confidence, holdMs, landmarks }) {
      const errors = [];
      if (!Array.isArray(landmarks) || landmarks.length < 21) {
        errors.push("no_hand");
        return { errors, hint: "Place one hand fully in frame.", pass: false };
      }
      if ((confidence || 0) < 0.6) errors.push("confidence_low");
      if ((holdMs || 0) < 1200) errors.push("hold_too_short");
      const indexTip = landmarks[8];
      const indexPip = landmarks[6];
      const middleTip = landmarks[12];
      const thumbTip = landmarks[4];
      const thumbIp = landmarks[3];
      const wrist = landmarks[0];
      if (tutorialDistance(indexTip, indexPip) > 0.08) errors.push("index_too_straight");
      if (tutorialDistance(indexTip, middleTip) > 0.07) errors.push("fingers_not_together");
      if (tutorialDistance(thumbTip, thumbIp) > 0.06) errors.push("thumb_too_far");
      if (Math.abs((wrist?.x || 0.5) - 0.5) > 0.35) errors.push("palm_too_rotated");
      const pass = predicted === target && errors.length <= 1;
      return { errors, hint: tutorialFallback(errors), pass };
    }

    function buildDetailedLocalAdvice({ target, predicted, errors = [], confidence, holdMs }) {
      const sections = [];
      if (predicted && target && predicted !== target) {
        sections.push(`Model sees ${predicted}, target is ${target}. Match ${target} by exaggerating its hand shape.`);
      }
      const shapeTips = [];
      if (errors.includes("index_too_straight")) shapeTips.push("Bend your index finger from the middle knuckle; avoid locking it straight.");
      if (errors.includes("fingers_not_together")) shapeTips.push("Press index and middle fingers closer so there is almost no gap.");
      if (errors.includes("thumb_too_far")) shapeTips.push("Slide your thumb in toward the index side so the thumb isn't flared out.");
      if (errors.includes("palm_too_rotated")) shapeTips.push("Turn your palm toward the camera about 10–20° until your fingertips face forward.");

      if (shapeTips.length) {
        sections.push(`Hand shape fixes:\n• ${shapeTips.join("\n• ")}`);
      }
      if (errors.includes("confidence_low") || errors.includes("no_hand")) {
        sections.push("Framing fix: keep your whole hand visible (wrist to fingertips), bring it closer to chest height, and use steady lighting.");
      }
      if (errors.includes("hold_too_short") || (holdMs || 0) < 1200) {
        sections.push("Timing fix: freeze the final hand shape for ~1.5 seconds before moving.");
      }
      if (!sections.length) {
        sections.push(`Refine ${target}: keep finger tension consistent, thumb controlled, and palm centered toward camera.`);
      }
      if ((confidence || 0) < 0.5) {
        sections.push("Extra: slow down your transition into the sign, then stop completely to help recognition lock in.");
      }
      return sections.join("\n\n");
    }

    async function requestTutorialAdvice(payload) {
      const key = `${payload.target}|${payload.predicted}|${[...payload.errors].sort().join(",")}`;
      if (tutorialAdviceCache.has(key)) return tutorialAdviceCache.get(key);
      try {
        const response = await fetch("/api/coach", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload, detailLevel: "detailed" }),
        });
        if (!response.ok) return { advice: tutorialFallback(payload.errors), options: [] };
        const body = await response.json();
        const advice = String(body?.advice || "").trim() || tutorialFallback(payload.errors);
        const options = Array.isArray(body?.options)
          ? body.options.filter((item) => typeof item === "string").slice(0, 3)
          : [];
        const result = { advice, options };
        tutorialAdviceCache.set(key, result);
        return result;
      } catch {
        return { advice: tutorialFallback(payload.errors), options: [] };
      }
    }

    function handleTutorialFeedback({ targetLetter, predicted, confidence, holdMs, landmarks }) {
      if (!targetLetter || !context.canSign()) return;
      const evaluated = evaluateTutorialSignals({
        target: targetLetter,
        predicted,
        confidence,
        holdMs,
        landmarks,
      });
      tutorialState.lastSignal = {
        target: targetLetter,
        predicted,
        confidence,
        holdMs,
        errors: evaluated.errors,
      };
      if (evaluated.pass) {
        tutorialState.failureByLetter.set(targetLetter, 0);
        setStatus(`Nice! ${targetLetter} looks good.`);
        setAdvice("Great effort—hold that shape and submit when ready.");
        return;
      }
      const failures = (tutorialState.failureByLetter.get(targetLetter) || 0) + 1;
      tutorialState.failureByLetter.set(targetLetter, failures);
      setStatus(`Working on ${targetLetter} · attempt ${failures}`);
      const chatClosed = ensureTutorialOverlay().chatPanel.hidden;
      setAdvice(chatClosed
        ? buildDetailedLocalAdvice({ target: targetLetter, predicted, errors: evaluated.errors, confidence, holdMs })
        : evaluated.hint);

      const nowMs = Date.now();
      if (failures >= 2 && nowMs - tutorialState.lastAdviceAt >= tutorialState.minAdviceIntervalMs) {
        tutorialState.lastAdviceAt = nowMs;
        void requestTutorialAdvice({
          target: targetLetter,
          predicted,
          confidence,
          holdMs,
          errors: evaluated.errors,
        }).then(({ advice, options }) => {
          const extra = options.length ? `\nTry next:\n• ${options.join("\n• ")}` : "";
          setAdvice(`${advice}${extra}`);
          if (options.length) {
            pushTutorialChat("assistant", `I noticed repeated trouble on ${targetLetter}. Here are options: ${options.join(" ")}`);
          }
        });
      }
    }

    function ensureTutorialOverlay() {
      if (tutorialOverlay) return tutorialOverlay;

      const root = document.createElement("section");
      root.dataset.tutorialOverlay = "true";
      root.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:9999;background:#111;color:#fff;padding:12px;border-radius:10px;max-width:320px;font:14px/1.4 sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)";

      const pillButton = document.createElement("button");
      pillButton.type = "button";
      pillButton.textContent = "Coach";
      pillButton.hidden = true;
      pillButton.style.cssText = "background:#2a6cf6;color:#fff;border:0;border-radius:12px 0 0 12px;padding:10px 12px;cursor:pointer;font-weight:600;letter-spacing:.2px;box-shadow:0 8px 24px rgba(0,0,0,.35)";

      const header = document.createElement("div");
      header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px";
      const title = document.createElement("h3");
      title.textContent = "Tutorial Coach";
      title.style.cssText = "margin:0;font-size:15px";

      const headerControls = document.createElement("div");
      headerControls.style.cssText = "display:flex;gap:6px";
      const minimizeButton = document.createElement("button");
      minimizeButton.type = "button";
      minimizeButton.textContent = "Minimize";
      minimizeButton.style.cssText = "background:#334155;color:#fff;border:0;border-radius:8px;padding:5px 8px;cursor:pointer;font-size:12px";
      headerControls.append(minimizeButton);
      header.append(title, headerControls);

      const content = document.createElement("div");
      content.style.cssText = "margin-top:6px";
      const status = document.createElement("p");
      status.style.cssText = "margin:4px 0";
      status.textContent = "Tutorial ready. Enable local ASL recognition to begin.";
      const advice = document.createElement("p");
      advice.style.cssText = "margin:4px 0;color:#cde;white-space:pre-line";
      advice.textContent = "Tip: Hold the expected sign steady for 1.5 seconds.";

      const toggleButton = document.createElement("button");
      toggleButton.type = "button";
      toggleButton.textContent = "Open chat";
      toggleButton.style.cssText = "margin-top:6px;background:#2a6cf6;color:#fff;border:0;border-radius:8px;padding:6px 10px;cursor:pointer";

      const chatPanel = document.createElement("div");
      chatPanel.hidden = true;
      chatPanel.style.cssText = "margin-top:8px;border-top:1px solid rgba(255,255,255,.2);padding-top:8px";

      const chatLog = document.createElement("div");
      chatLog.style.cssText = "max-height:180px;overflow:auto;padding-right:4px";

      const composer = document.createElement("div");
      composer.style.cssText = "display:flex;gap:6px;margin-top:8px";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Ask the coach anything…";
      input.maxLength = 240;
      input.style.cssText = "flex:1;border-radius:8px;border:1px solid #555;background:#1c1c1c;color:#fff;padding:6px 8px";

      const sendButton = document.createElement("button");
      sendButton.type = "button";
      sendButton.textContent = "Send";
      sendButton.style.cssText = "background:#2a6cf6;color:#fff;border:0;border-radius:8px;padding:6px 10px;cursor:pointer";
      composer.append(input, sendButton);

      const attachmentsRow = document.createElement("div");
      attachmentsRow.style.cssText = "display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap";
      const attachInput = document.createElement("input");
      attachInput.type = "file";
      attachInput.accept = "image/png,image/jpeg,image/webp";
      attachInput.hidden = true;

      const attachButton = document.createElement("button");
      attachButton.type = "button";
      attachButton.textContent = "Attach photo";
      attachButton.style.cssText = "background:#334155;color:#fff;border:0;border-radius:8px;padding:6px 8px;cursor:pointer";

      const generateImageButton = document.createElement("button");
      generateImageButton.type = "button";
      generateImageButton.textContent = "Get letter image";
      generateImageButton.style.cssText = "background:#334155;color:#fff;border:0;border-radius:8px;padding:6px 8px;cursor:pointer";

      const attachmentStatus = document.createElement("span");
      attachmentStatus.style.cssText = "font-size:12px;color:#cde";
      attachmentStatus.textContent = "No photo attached";
      attachmentsRow.append(attachInput, attachButton, generateImageButton, attachmentStatus);

      chatPanel.append(chatLog, composer, attachmentsRow);
      content.append(status, advice, toggleButton, chatPanel);
      root.append(pillButton, header, content);
      document.body.appendChild(root);

      tutorialOverlay = {
        root,
        header,
        content,
        pillButton,
        status,
        advice,
        toggleButton,
        minimizeButton,
        chatPanel,
        chatLog,
        input,
        sendButton,
        attachInput,
        attachButton,
        generateImageButton,
        attachmentStatus,
      };

      toggleButton.addEventListener("click", () => toggleTutorialChat());
      minimizeButton.addEventListener("click", () => toggleTutorialMinimized());
      pillButton.addEventListener("click", () => setTutorialMinimized(false));
      sendButton.addEventListener("click", () => { void sendTutorialChatMessage(); });
      attachButton.addEventListener("click", () => attachInput.click());
      generateImageButton.addEventListener("click", () => {
        const suggested = tutorialState.lastSignal?.target || "";
        const letter = prompt("Which letter do you want a reference image for?", suggested);
        if (!letter) return;
        void requestTutorialLetterImage(letter);
      });
      attachInput.addEventListener("change", () => {
        const file = attachInput.files?.[0];
        if (!file) return;
        if (file.size > 2 * 1024 * 1024) {
          attachmentStatus.textContent = "Photo too large (max 2MB).";
          attachInput.value = "";
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          if (typeof reader.result !== "string") return;
          tutorialState.pendingPhoto = { dataUrl: reader.result, name: file.name };
          attachmentStatus.textContent = `Attached: ${file.name}`;
          void sendTutorialChatMessage("Please review this hand photo and give detailed corrections.", { autoPhoto: true });
        };
        reader.readAsDataURL(file);
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          void sendTutorialChatMessage();
        }
      });

      setTutorialMinimized(false);
      pushTutorialChat("assistant", "Hey! I’m your ASL tutorial coach. Open a chat and ask me how to improve a letter.");
      return tutorialOverlay;
    }

    function setStatus(text) {
      ensureTutorialOverlay().status.textContent = text;
    }

    function setAdvice(text) {
      ensureTutorialOverlay().advice.textContent = text;
    }

    return {
      ensureOverlay: ensureTutorialOverlay,
      setStatus,
      setAdvice,
      handleFeedback: handleTutorialFeedback,
    };
  }

  window.createTutorialCoach = createTutorialCoach;
})();
