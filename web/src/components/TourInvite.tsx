import { useAppStore } from "../state/store";
import { FLAGSHIP_TOUR_ID, tourById } from "../tours/registry";

/**
 * A one-time offer of a tour, for a reader who has never been shown one.
 *
 * It sits in a row in the shell rather than a card over it, and never a modal.
 * Anyone who already knows this instrument should be able to ignore it
 * entirely and use what is underneath, which is the whole point of the skip.
 * The answer is remembered either way, so it appears once per browser and then
 * never again.
 *
 * The title and blurb are the tour's own. A second description written here
 * would be a claim about the tour that nothing checks, and the registry test
 * already holds the tour's own prose to the rules.
 */
export function TourInvite() {
  const inviteOpen = useAppStore((s) => s.inviteOpen);
  const tourId = useAppStore((s) => s.tourId);
  const startTour = useAppStore((s) => s.startTour);
  const dismissInvite = useAppStore((s) => s.dismissInvite);
  const tour = tourById(FLAGSHIP_TOUR_ID);
  // Stay away while a tour is running: a deep link straight into one arrives
  // with the invitation still unanswered for a beat.
  if (!inviteOpen || tourId || !tour) return null;
  return (
    <aside className="tour-invite" aria-label="guided tour invitation">
      <p className="tour-invite-text">
        <span className="tour-invite-lead">New here?</span> {tour.title}: {tour.blurb}{" "}
        <span className="tour-invite-count">{tour.steps.length} steps.</span>
      </p>
      <div className="tour-invite-actions">
        <button
          type="button"
          className="tour-invite-take"
          onClick={() => startTour(FLAGSHIP_TOUR_ID, 0)}
        >
          take the tour
        </button>
        <button type="button" className="link-button" onClick={dismissInvite}>
          skip to the app
        </button>
      </div>
      <p className="tour-invite-note">
        Skip and this stays out of the way for good. The tours live in the top bar.
      </p>
    </aside>
  );
}
