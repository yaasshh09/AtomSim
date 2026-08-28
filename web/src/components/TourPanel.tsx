import { useAppStore } from "../state/store";
import { tourById } from "../tours/registry";

/**
 * The tour's narration, which I dock under the stage.
 *
 * I dock it rather than floating it over the views: the picture is the thing
 * the prose is about, and a card on top of it would cover the evidence.
 */
export function TourPanel() {
  const { tourId, stepIndex, goToStep, exitTour, finishTour } = useAppStore();
  const tour = tourId ? tourById(tourId) : null;
  if (!tour) return null;
  const step = tour.steps[stepIndex];
  if (!step) return null;
  const last = tour.steps.length - 1;
  return (
    <aside className="tour-panel" aria-label={`${tour.title}, step ${stepIndex + 1}`}>
      <div className="tour-head">
        <span className="tour-count">
          {stepIndex + 1} / {tour.steps.length}
        </span>
        <span className="tour-title">{step.title}</span>
        <button className="tour-close" type="button" onClick={exitTour} aria-label="leave the tour">
          ✕
        </button>
      </div>
      {step.body.map((p, i) => (
        <p key={i} className="tour-body">
          {p}
        </p>
      ))}
      <div className="tour-nav">
        <button
          type="button"
          className="link-button"
          onClick={() => goToStep(stepIndex - 1)}
          disabled={stepIndex === 0}
        >
          ‹ back
        </button>
        {stepIndex === last ? (
          // I call finishTour, not exitTour: reaching the end is the one fact
          // worth remembering about a tour, and it is what I mark done in the
          // menu.
          <button type="button" className="link-button" onClick={finishTour}>
            finish ›
          </button>
        ) : (
          <button type="button" className="link-button" onClick={() => goToStep(stepIndex + 1)}>
            next ›
          </button>
        )}
      </div>
    </aside>
  );
}
