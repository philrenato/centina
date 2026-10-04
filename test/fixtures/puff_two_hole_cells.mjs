// Two-hole puffs traced from drawings on the ground plane. Control points are screen pixels of a
// perspective view; `ground` undoes the view approximately (y stretched, the far side widened) and
// `sample` evaluates the closed cubic B-spline the points control, as the app's curves do.
//
// notch: a blobby outline with a deep rounded notch at the top center and an indent on the right;
//   a small bean at lower left and a large tilted oval right of center, below the notch. The oval's
//   power-diagram cell takes both walls of the notch and is not star-shaped about the oval.
// kidney: a rounded outline with a lobe at the lower right; a rounded left hole with a dent at its
//   bottom and a right hole shaped like a kidney, its deep notch opening to the right.

export const NOTCH = {
  outer: [[820, 107], [960, 77], [1056, 77], [1101, 98], [1145, 161], [1130, 215], [1174, 268], [1266, 306], [1340, 312], [1400, 290], [1430, 215], [1470, 155], [1560, 115], [1695, 79], [1805, 77], [1850, 90], [1885, 130], [1898, 179], [1908, 270], [1930, 320], [1907, 365], [1860, 410], [1887, 460], [1910, 490], [1900, 520], [1825, 558], [1760, 600], [1711, 715], [1562, 879], [1290, 1020], [1130, 1034], [841, 1033], [722, 1002], [600, 900], [527, 790], [520, 600], [557, 432], [650, 300], [731, 177]],
  holes: [
    [[930, 633], [962, 620], [1015, 618], [1042, 630], [1068, 653], [1072, 683], [1063, 692], [1037, 704], [991, 714], [948, 725], [924, 742], [904, 742], [882, 686], [900, 659]],
    [[1493, 332], [1537, 335], [1560, 345], [1577, 360], [1590, 390], [1601, 405], [1602, 440], [1584, 467], [1565, 488], [1532, 527], [1482, 583], [1447, 612], [1418, 630], [1382, 637], [1337, 637], [1312, 616], [1301, 587], [1298, 528], [1313, 482], [1345, 433], [1407, 371], [1456, 343]],
  ],
  view: { cx: 1200, cy: 560, near: 1034, depth: 957, sy: 1.6, persp: 0.3 },
};

export const KIDNEY = {
  outer: [[1258, 77], [1308, 76], [1392, 82], [1466, 107], [1505, 126], [1541, 146], [1594, 183], [1639, 225], [1665, 255], [1693, 297], [1721, 346], [1737, 390], [1742, 462], [1736, 488], [1696, 585], [1666, 633], [1628, 682], [1557, 764], [1493, 840], [1460, 904], [1424, 935], [1405, 942], [1326, 957], [1037, 870], [898, 767], [818, 596], [813, 453], [865, 341], [1017, 170], [1108, 112]],
  holes: [
    [[1095, 487], [1135, 487], [1155, 500], [1163, 507], [1188, 530], [1206, 555], [1214, 585], [1214, 609], [1210, 620], [1201, 630], [1187, 643], [1168, 655], [1157, 658], [1142, 657], [1121, 648], [1107, 652], [1105, 662], [1075, 662], [1068, 659], [1048, 652], [1043, 645], [1035, 632], [1030, 608], [1030, 580], [1033, 563], [1044, 535], [1060, 511], [1080, 493], [1110, 486]],
    [[1415, 325], [1449, 316], [1482, 312], [1514, 312], [1532, 325], [1548, 340], [1568, 368], [1576, 386], [1578, 413], [1576, 426], [1565, 448], [1553, 465], [1530, 484], [1527, 500], [1531, 506], [1570, 526], [1593, 545], [1598, 561], [1585, 586], [1568, 611], [1548, 620], [1530, 622], [1471, 621], [1450, 614], [1426, 598], [1412, 578], [1380, 520], [1361, 476], [1351, 435], [1352, 396], [1372, 354], [1391, 338]],
  ],
  view: { cx: 1280, cy: 520, near: 960, depth: 880, sy: 1.3, persp: 0.2 },
};

export function ground(P, v, sy = v.sy, persp = v.persp) {
  return P.map(([x, y]) => { const w = 1 + persp * (v.near - y) / v.depth; return [(x - v.cx) * w, -(y - v.cy) * sy * w]; });
}

// 128 samples evenly in the parameter, as the app's Puff reads a curve.
export function sample(P, n = 128) {
  const out = [], m = P.length;
  for (let s = 0; s < n; s++) {
    const u = (s / n) * m, i = Math.floor(u), t = u - i, t2 = t * t, t3 = t2 * t;
    const p0 = P[(i - 1 + m) % m], p1 = P[i % m], p2 = P[(i + 1) % m], p3 = P[(i + 2) % m];
    const b0 = (1 - t) ** 3 / 6, b1 = (3 * t3 - 6 * t2 + 4) / 6, b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, b3 = t3 / 6;
    out.push([b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0], b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]]);
  }
  return out;
}

// One case in the plane: the outline and holes as sampled loops. `sy`/`persp` vary the unprojection.
export function loopsOf(fx, sy, persp) {
  const g = (P) => sample(ground(P, fx.view, sy, persp));
  return { outer: g(fx.outer), holes: fx.holes.map(g) };
}

// A hole moved and turned about its own centroid: dx, dy as fractions of the outline's span.
export function jitterHole(loop, span, dx, dy, ang) {
  let cx = 0, cy = 0; for (const p of loop) { cx += p[0]; cy += p[1]; } cx /= loop.length; cy /= loop.length;
  const c = Math.cos(ang), s = Math.sin(ang);
  return loop.map(([x, y]) => { const u = x - cx, w = y - cy; return [cx + c * u - s * w + dx * span, cy + s * u + c * w + dy * span]; });
}

export function spanOf(loop) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of loop) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return Math.max(x1 - x0, y1 - y0);
}
