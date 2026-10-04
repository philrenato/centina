// An Infill piped as the app pipes it (runInfillPipe): a radius of 0.12 of the strut length to
// three significant figures, six facets, two segments, hull joints, radii fitted per rail.
import { subdPipeNetwork } from '../../kernel/subdnetwork.mjs';
import { subdivideCatmullClark } from '../../kernel/subd.mjs';
import { extrudePolygon, orientOutward } from './infill_bodies.mjs';

export const box = () => orientOutward(extrudePolygon([[0, 0], [100, 0], [100, 100], [0, 100]], 100));
export const lshape = () => orientOutward(extrudePolygon([[0, 0], [100, 0], [100, 35], [35, 35], [35, 100], [0, 100]], 49));
export function pipeInfill(res) {
  const rails = res.curves.map((c) => ({ degree: 3, knots: c.curve.knots, ctrlPts: c.curve.ctrlPts.map((p) => [...p, 1]) }));
  const radius = Number((res.report.strutLength * 0.12).toPrecision(3));
  return subdPipeNetwork(rails, { radius, facets: 6, segments: 2, crease: 0, junction: 'hull', fitRadius: true, weld: true, weldFraction: 0.5, capStart: 'round', capEnd: 'round' });
}
export const limitOf = (c) => subdivideCatmullClark(subdivideCatmullClark({ vertices: c.vertices, faces: c.faces, creases: c.creases || {} }));
