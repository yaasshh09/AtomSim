/**
 * Getting a plot out of the page: as an SVG, a PNG, or the numbers behind it.
 *
 * The plots are styled by class from index.css, which a saved .svg would not
 * carry, so the export copies each element's computed paint onto the clone.
 * What is saved is therefore exactly what is on screen, in the current theme,
 * at the current zoom.
 */

const PAINT = [
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-width",
  "stroke-dasharray",
  "stroke-opacity",
  "opacity",
  "display",
  "visibility",
  "paint-order",
  "font-family",
  "font-size",
  "font-weight",
  "font-style",
  "text-anchor",
  "dominant-baseline",
] as const;

/** Whether a computed colour is fully opaque, in whatever syntax the browser
 *  returns it (rgb(), color(srgb ...), oklab...): paint it and read alpha. */
function isOpaque(color: string): boolean {
  const ctx = document.createElement("canvas").getContext("2d");
  if (!ctx) return false;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  return ctx.getImageData(0, 0, 1, 1).data[3] === 255;
}

/** The first opaque background up the tree, so the export is not see-through.
 *  Translucent washes (the panels use some) are skipped, not composited. */
function backdrop(el: Element): string {
  for (let e: Element | null = el; e; e = e.parentElement) {
    const bg = getComputedStyle(e).backgroundColor;
    if (bg && isOpaque(bg)) return bg;
  }
  return "white";
}

/** A standalone copy of the plot, with its styles inlined and (unless it is an
 *  overlay for a raster) an opaque background. */
export function serializePlot(
  svg: SVGSVGElement,
  withBackground = true,
): { text: string; w: number; h: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const src = [svg, ...svg.querySelectorAll("*")];
  const dst = [clone, ...clone.querySelectorAll("*")];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const target = dst[i] as SVGElement;
    target.removeAttribute("class");
    target.setAttribute("style", PAINT.map((p) => `${p}:${cs.getPropertyValue(p)}`).join(";"));
  });
  const [, , w, h] = (svg.getAttribute("viewBox") ?? "0 0 800 400").split(/\s+/).map(Number);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  if (withBackground) {
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("width", "100%");
    bg.setAttribute("height", "100%");
    bg.setAttribute("fill", backdrop(svg));
    clone.insertBefore(bg, clone.firstChild);
  }
  return { text: new XMLSerializer().serializeToString(clone), w, h };
}

export function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function saveSvg(svg: SVGSVGElement, name: string) {
  download(`${name}.svg`, new Blob([serializePlot(svg).text], { type: "image/svg+xml" }));
}

/** Rendered at twice the plot's size, which holds up in a slide or a paper. */
export function savePng(svg: SVGSVGElement, name: string, scale = 2) {
  const { text, w, h } = serializePlot(svg);
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = w * scale;
    canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    canvas.toBlob((b) => b && download(`${name}.png`, b), "image/png");
  };
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(text)}`;
}

/** RFC 4180 CSV: fields with a comma, quote or newline are quoted. */
export function toCsv(header: readonly string[], rows: readonly (readonly unknown[])[]): string {
  const cell = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header, ...rows].map((r) => r.map(cell).join(",")).join("\n") + "\n";
}

/**
 * A raster plot as a PNG: the image scaled up to a usable size, with the grid
 * overlay (if any) drawn on top. Smoothing matches the on-screen picture.
 */
export function saveCanvasPng(
  canvas: HTMLCanvasElement,
  overlay: SVGSVGElement | null,
  name: string,
  size = 1024,
) {
  // The longer side gets `size`; the 3-D stage is not square.
  const k = size / Math.max(canvas.width, canvas.height, 1);
  const w = Math.round(canvas.width * k);
  const h = Math.round(canvas.height * k);
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(canvas, 0, 0, w, h);
  const finish = () => out.toBlob((b) => b && download(`${name}.png`, b), "image/png");
  if (!overlay) return finish();
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, w, h);
    finish();
  };
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
    serializePlot(overlay, false).text,
  )}`;
}

/** A file name from the plot's own title: "R(r), the radial..." -> "r-r-the-radial". */
export function plotFileName(svg: SVGSVGElement | null): string {
  const lead =
    svg?.closest("figure, .zoom-panel, .view-wrap")?.querySelector(".plot-lead, .plot-title")
      ?.textContent ?? "";
  const slug = lead
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
  return `atomsim-${slug || "plot"}`;
}
