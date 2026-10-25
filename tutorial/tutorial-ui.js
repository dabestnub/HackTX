function createTutorialOverlay(options = {}) {
  const doc = options.document || globalThis.document;
  if (!doc) throw new Error("A document is required to render tutorial UI");

  const root = doc.createElement("section");
  root.setAttribute("data-tutorial-overlay", "true");
  root.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:9999;background:#111;color:#fff;padding:12px;border-radius:10px;max-width:320px;font:14px/1.4 sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)";

  const title = doc.createElement("h3");
  title.textContent = "Tutorial Coach";
  title.style.cssText = "margin:0 0 6px;font-size:15px";

  const status = doc.createElement("p");
  status.textContent = "Tutorial is ready.";
  status.style.cssText = "margin:4px 0";

  const advice = doc.createElement("p");
  advice.textContent = "Tip: Hold the letter shape steady.";
  advice.style.cssText = "margin:4px 0;color:#cde";

  root.append(title, status, advice);

  return {
    mount(parent = doc.body) {
      parent.appendChild(root);
    },
    unmount() {
      if (root.parentNode) root.parentNode.removeChild(root);
    },
    setStatus(text) {
      status.textContent = text || "";
    },
    setAdvice(text) {
      advice.textContent = text || "";
    },
    element: root,
  };
}

module.exports = { createTutorialOverlay };
