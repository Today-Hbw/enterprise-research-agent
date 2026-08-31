(function exposeMarkdown(globalScope) {
  "use strict";

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function safeHttpUrl(value) {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
    } catch (_error) {
      return null;
    }
  }

  function formatEscapedInline(value) {
    return value
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  }

  function renderInline(value) {
    const code = [];
    const links = [];
    let prepared = String(value).replace(/`([^`\n]+)`/g, (_match, content) => {
      const token = `\uE000CODE${code.length}\uE001`;
      code.push(`<code>${escapeHtml(content)}</code>`);
      return token;
    });
    prepared = prepared.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
      const safeUrl = safeHttpUrl(href);
      if (!safeUrl) return match;
      const token = `\uE000LINK${links.length}\uE001`;
      links.push(
        `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noreferrer noopener">${formatEscapedInline(escapeHtml(label))}</a>`,
      );
      return token;
    });

    let rendered = formatEscapedInline(escapeHtml(prepared));
    rendered = rendered.replace(
      /\uE000CODE(\d+)\uE001/g,
      (match, index) => code[Number(index)] ?? match,
    );
    return rendered.replace(
      /\uE000LINK(\d+)\uE001/g,
      (match, index) => links[Number(index)] ?? match,
    );
  }

  function renderMarkdown(value) {
    const lines = String(value).replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n");
    const output = [];
    let paragraph = [];
    let listType = null;
    let listItems = [];
    let fence = null;
    let codeLines = [];

    function flushParagraph() {
      if (!paragraph.length) return;
      output.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
      paragraph = [];
    }

    function flushList() {
      if (!listType) return;
      output.push(`<${listType}>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
      listType = null;
      listItems = [];
    }

    function flushBlocks() {
      flushParagraph();
      flushList();
    }

    for (const line of lines) {
      if (fence !== null) {
        if (/^```\s*$/.test(line)) {
          const language = fence ? ` class="language-${escapeHtml(fence)}"` : "";
          output.push(`<pre><code${language}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
          fence = null;
          codeLines = [];
        } else {
          codeLines.push(line);
        }
        continue;
      }

      const fenceMatch = line.match(/^```([a-zA-Z0-9_-]*)\s*$/);
      if (fenceMatch) {
        flushBlocks();
        fence = fenceMatch[1];
        continue;
      }
      if (!line.trim()) {
        flushBlocks();
        continue;
      }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        flushBlocks();
        const level = heading[1].length;
        output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }
      if (/^\s*(---|___)\s*$/.test(line)) {
        flushBlocks();
        output.push("<hr>");
        continue;
      }
      const quote = line.match(/^>\s?(.*)$/);
      if (quote) {
        flushBlocks();
        output.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
        continue;
      }

      const unordered = line.match(/^\s*[-*]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
      const nextListType = unordered ? "ul" : ordered ? "ol" : null;
      if (nextListType) {
        flushParagraph();
        if (listType && listType !== nextListType) flushList();
        listType = nextListType;
        listItems.push((unordered || ordered)[1]);
        continue;
      }

      flushList();
      paragraph.push(line.trim());
    }

    if (fence !== null) {
      const language = fence ? ` class="language-${escapeHtml(fence)}"` : "";
      output.push(`<pre><code${language}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    }
    flushBlocks();
    return output.join("");
  }

  const api = { renderMarkdown };
  globalScope.ResearchMarkdown = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
