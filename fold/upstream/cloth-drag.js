import { Plane, Raycaster, Vector2, Vector3 } from 'three';

export function addClothDragging(canvas, camera, sheet) {
  const raycaster = new Raycaster();
  const pointer = new Vector2();
  const plane = new Plane();
  const normal = new Vector3(), point = new Vector3(), offset = new Vector3();
  let pointerId = null;
  const sim = sheet.simulation;

  function setRay(event, clampToCanvas = false) {
    const bounds = canvas.getBoundingClientRect();
    pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, 1 - (event.clientY - bounds.top) / bounds.height * 2);
    if (clampToCanvas) pointer.clampScalar(-0.99, 0.99);
    camera.updateMatrixWorld();
    raycaster.setFromCamera(pointer, camera);
  }

  function cancel() {
    const id = pointerId;
    pointerId = null;
    sim.endGrab();
    canvas.classList.remove('grabbing');
    if (id !== null && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
  }

  function down(event) {
    if (pointerId !== null || !event.isPrimary || event.button !== 0) return;
    setRay(event);
    sheet.mesh.updateMatrixWorld();
    // The mesh deforms every frame, so refresh the raycast broad-phase bounds.
    sheet.geometry.computeBoundingSphere();
    sheet.geometry.computeBoundingBox();
    const hits = raycaster.intersectObject(sheet.mesh, false);
    for (const hit of hits) {
      if (!hit.uv || hit.point.y < 0) continue;
      const row = Math.round((1 - hit.uv.y) * (sim.activeRows - 1));
      const col = Math.max(0, Math.min(sim.columns, Math.round(hit.uv.x * sim.columns)));
      const index = row * sim.stride + col;
      if (!sim.beginGrab(index)) continue;
      camera.getWorldDirection(normal);
      plane.setFromNormalAndCoplanarPoint(normal, hit.point);
      offset.fromArray(sim.positions, index * 3).sub(hit.point);
      pointerId = event.pointerId;
      canvas.setPointerCapture(pointerId);
      canvas.classList.add('grabbing');
      event.preventDefault();
      break;
    }
  }

  function move(event) {
    if (event.pointerId !== pointerId) return;
    if (!sim.grab) { cancel(); return; }
    setRay(event, true);
    if (raycaster.ray.intersectPlane(plane, point)) sim.moveGrab(point.add(offset).toArray());
    event.preventDefault();
  }

  function up(event) {
    if (event.pointerId === pointerId) cancel();
  }

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('lostpointercapture', up);
  window.addEventListener('blur', cancel);
  return { cancel, get active() { return pointerId !== null; } };
}
