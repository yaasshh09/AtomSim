import { useRef } from "react";
import { GLOSSARY, glossParts } from "../lib/glossary";
import { Notation } from "../lib/mathText";

/**
 * Running text with its glossary terms marked: the first mention of each gets a
 * dotted underline and its plain-language definition on hover or focus.
 */
export function Glossed({ children }: { children: string }) {
  return (
    <>
      {glossParts(children).map((p, i) =>
        p.term ? (
          <abbr key={i} className="gloss" title={GLOSSARY[p.term]} tabIndex={0}>
            <Notation>{p.text}</Notation>
          </abbr>
        ) : (
          <Notation key={i}>{p.text}</Notation>
        ),
      )}
    </>
  );
}

/** Every term and its definition, in a native dialog opened from the rail's footer. */
export function GlossaryLink() {
  const ref = useRef<HTMLDialogElement>(null);
  const terms = Object.keys(GLOSSARY).sort((a, b) => a.localeCompare(b));
  return (
    <>
      <button type="button" className="link-button" onClick={() => ref.current?.showModal()}>
        Glossary
      </button>
      <dialog ref={ref} className="glossary-dialog" aria-labelledby="glossary-title">
        <h2 id="glossary-title">Glossary</h2>
        <dl>
          {terms.map((t) => (
            <div key={t} className="glossary-row">
              <dt>
                <Notation>{t}</Notation>
              </dt>
              <dd>
                <Notation>{GLOSSARY[t]}</Notation>
              </dd>
            </div>
          ))}
        </dl>
        <form method="dialog">
          <button type="submit" className="ghost">
            Close
          </button>
        </form>
      </dialog>
    </>
  );
}
