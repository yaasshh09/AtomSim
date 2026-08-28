import { useAppStore } from "../state/store";
import { FLAGSHIP_TOUR_ID, tourById } from "../tours/registry";

/**
 * My one-time offer of a tour, for a reader I have never shown one to.
 *
 * I put it in a row in the shell rather than a card over it, and never a
 * modal: if you already know this instrument you should be able to ignore me
 * entirely and use what is underneath, which is the whole point of the skip. I
 * remember your answer either way, so I appear once per browser and then never
 * again.
 *
 * The title and blurb are the tour's own. If I wrote a second description here
 * it would be a claim about the tour that nothing checks, and my registry test
 * already holds the tour's own prose to my rules.
 */
export function TourInvite() {
  const inviteOpen = useAppStore((s) => s.inviteOpen);
  const tourId = useAppStore((s) => s.tourId);
  const startTour = useAppStore((s) => s.startTour);
  const dismissInvite = useAppStore((s) => s.dismissInvite);
  const tour = tourById(FLAGSHIP_TOUR_ID);
  // I stay away while a tour is running: a deep link straight into one arrives
  // with my invitation still unanswered for a beat.
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
        Skip and I keep this out of your way for good. I leave the tours in the top bar.
      </p>
    </aside>
  );
}
