/* Render every non-code occurrence of the word "bifrost" as "bifröst" in regular Eczar (.bf-word). */
(() => {
  const SKIP =
    "code,pre,kbd,samp,var,script,style,noscript,template,textarea,svg,math,[contenteditable],.highlight,.arithmatex,.mjx-container,.bf-word,.md-source";
  const NON_TEXT_SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "TEXTAREA", "SVG"]);
  const OBSERVE_OPTS = { childList: true, subtree: true, characterData: true };
  const BLOCKS = new Set([
    "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "BR", "DD", "DETAILS", "DIV", "DL", "DT",
    "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3", "H4", "H5", "H6",
    "HEADER", "HGROUP", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION",
    "SUMMARY", "TABLE", "TD", "TH", "TR", "UL",
  ]);
  const WORD = /(?<![\p{L}\p{N}_/@-])(?<![\p{L}\p{N}_]\.)bifr[o\u00f6]st(?![\p{L}\p{N}_/-]|\.[\p{L}\p{N}_])/giu;

  let observer = null;
  let mutating = false;

  function skipMarker(node) {
    if (NON_TEXT_SKIP.has(node.tagName.toUpperCase())) return " ";
    const t = node.textContent || "";
    if (!t) return " ";
    const edge = t.length <= 4 ? t : t.slice(0, 2) + " " + t.slice(-2);
    return edge.replace(/[\p{L}\p{N}_]/gu, "x");
  }

  function withoutObserving(fn) {
    if (mutating) return fn();
    mutating = true;
    if (observer) observer.disconnect();
    try {
      return fn();
    } finally {
      if (observer && document.body) observer.observe(document.body, OBSERVE_OPTS);
      mutating = false;
    }
  }

  function blockAncestor(node) {
    let cur = node instanceof Element ? node : node.parentElement;
    while (cur && cur !== document.body && !BLOCKS.has(cur.tagName)) {
      cur = cur.parentElement;
    }
    return cur || document.body;
  }

  function styleRoot(root) {
    if (!root || root.nodeType !== Node.ELEMENT_NODE) return;
    if (root.matches(SKIP) || root.closest(SKIP)) return;

    let buf = "";
    let segs = [];

    function flush() {
      if (!segs.length) return;
      if (/bifr[o\u00f6]st/i.test(buf)) {
        WORD.lastIndex = 0;
        const matches = [];
        let m;
        while ((m = WORD.exec(buf)) !== null) {
          matches.push([m.index, m.index + m[0].length]);
        }
        if (matches.length) {
          const edits = new Map();
          for (const [mStart, mEnd] of matches) {
            const vowelIdx = mStart + 4;
            for (const seg of segs) {
              if (seg.end <= mStart) continue;
              if (seg.start >= mEnd) break;
              const lo = Math.max(mStart, seg.start);
              const hi = Math.min(mEnd, seg.end);
              const rStart = lo - seg.start;
              const rEnd = hi - seg.start;
              let word = seg.node.nodeValue.slice(rStart, rEnd);
              if (lo <= vowelIdx && vowelIdx < hi) {
                const at = vowelIdx - lo;
                const ch = word[at];
                const repl = ch === "o" ? "\u00f6" : ch === "O" ? "\u00d6" : ch;
                if (repl !== ch) word = word.slice(0, at) + repl + word.slice(at + 1);
              }
              let list = edits.get(seg.node);
              if (!list) {
                list = [];
                edits.set(seg.node, list);
              }
              list.push([rStart, rEnd, word]);
            }
          }
          if (edits.size) {
            withoutObserving(() => {
              for (const [node, ranges] of edits) {
                const text = node.nodeValue;
                const frag = document.createDocumentFragment();
                let pos = 0;
                for (const [start, end, word] of ranges) {
                  if (start > pos) frag.append(document.createTextNode(text.slice(pos, start)));
                  const span = document.createElement("span");
                  span.className = "bf-word";
                  span.textContent = word;
                  frag.append(span);
                  pos = end;
                }
                if (pos < text.length) frag.append(document.createTextNode(text.slice(pos)));
                node.replaceWith(frag);
              }
            });
          }
        }
      }
      buf = "";
      segs = [];
    }

    function walk(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        const v = node.nodeValue;
        if (v.length) {
          segs.push({ node, start: buf.length, end: buf.length + v.length });
          buf += v;
        }
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const isBlock = BLOCKS.has(node.tagName);
      if (isBlock) flush();
      if (node.matches(SKIP)) {
        if (!isBlock) buf += skipMarker(node);
        return;
      }
      for (const child of [...node.childNodes]) walk(child);
      if (isBlock) flush();
    }

    walk(root);
    flush();
  }

  function applyAll() {
    if (document.body) withoutObserving(() => styleRoot(document.body));
  }

  function boot() {
    applyAll();
    if (!observer && document.body) {
      observer = new MutationObserver((records) => {
        if (mutating) return;
        const roots = new Set();
        for (const r of records) {
          if (r.type === "characterData") {
            const p = r.target.parentElement;
            if (p && p.isConnected && !p.closest(SKIP)) roots.add(blockAncestor(p));
          } else {
            for (const n of r.addedNodes) {
              if (n.nodeType === Node.ELEMENT_NODE) {
                if (!n.matches(SKIP) && !n.closest(SKIP)) roots.add(blockAncestor(n));
              } else if (n.nodeType === Node.TEXT_NODE) {
                const p = r.target instanceof Element ? r.target : r.target.parentElement;
                if (p && !p.closest(SKIP)) roots.add(blockAncestor(p));
              }
            }
          }
        }
        if (!roots.size) return;
        withoutObserving(() => {
          for (const root of roots) {
            if (root.isConnected) styleRoot(root);
          }
        });
      });
      observer.observe(document.body, OBSERVE_OPTS);
    }
    if (typeof document$ !== "undefined" && document$ && typeof document$.subscribe === "function") {
      document$.subscribe(applyAll);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
