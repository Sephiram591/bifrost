/* Style every non-code occurrence of the word "bifrost" in regular Eczar (.bf-word). */
(() => {
  const SKIP =
    "code,pre,kbd,samp,var,script,style,noscript,template,textarea,svg,math,.highlight,.arithmatex,.mjx-container,.bf-word,.md-header__topic:first-child,.md-nav__title[for='__drawer'],.md-source";
  const BLOCKS = new Set([
    "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "BR", "DD", "DETAILS", "DIV", "DL", "DT",
    "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3", "H4", "H5", "H6",
    "HEADER", "HGROUP", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION",
    "SUMMARY", "TABLE", "TD", "TH", "TR", "UL",
  ]);
  const WORD = /(?<![\w/@-])(?<!\w\.)bifrost(?![\w/-]|\.\w)/gi;

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
      if (/bifrost/i.test(buf)) {
        WORD.lastIndex = 0;
        const matches = [];
        let m;
        while ((m = WORD.exec(buf)) !== null) {
          matches.push([m.index, m.index + m[0].length]);
        }
        if (matches.length) {
          const edits = new Map();
          for (const [mStart, mEnd] of matches) {
            for (const seg of segs) {
              if (seg.end <= mStart) continue;
              if (seg.start >= mEnd) break;
              const lo = Math.max(mStart, seg.start);
              const hi = Math.min(mEnd, seg.end);
              let list = edits.get(seg.node);
              if (!list) {
                list = [];
                edits.set(seg.node, list);
              }
              list.push([lo - seg.start, hi - seg.start]);
            }
          }
          for (const [node, ranges] of edits) {
            const text = node.nodeValue;
            const frag = document.createDocumentFragment();
            let pos = 0;
            for (const [start, end] of ranges) {
              if (start > pos) frag.append(document.createTextNode(text.slice(pos, start)));
              const span = document.createElement("span");
              span.className = "bf-word";
              span.textContent = text.slice(start, end);
              frag.append(span);
              pos = end;
            }
            if (pos < text.length) frag.append(document.createTextNode(text.slice(pos)));
            node.replaceWith(frag);
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
      if (node.matches(SKIP)) {
        buf += " ";
        return;
      }
      const isBlock = BLOCKS.has(node.tagName);
      if (isBlock) flush();
      for (const child of [...node.childNodes]) walk(child);
      if (isBlock) flush();
    }

    walk(root);
    flush();
  }

  function applyAll() {
    if (document.body) styleRoot(document.body);
  }

  let observing = false;
  function boot() {
    applyAll();
    if (!observing && document.body) {
      observing = true;
      const observer = new MutationObserver((records) => {
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
        for (const root of roots) {
          if (root.isConnected) styleRoot(root);
        }
      });
      observer.observe(document.body, { childList: true, subtree: true, characterData: true });
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
