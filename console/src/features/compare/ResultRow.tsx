import { motion } from "motion/react";
import type { Listing } from "../../core/types";
import { condition, eur } from "../../shared/format";
import { move } from "../../theme/motion";

/** One listing as a hang tag: the seller's own title, what the seller filled in, the price. */
export function ResultRow({ item, rank, moved }: { item: Listing; rank: number; moved: number | null }) {
  const facts = [item.brand, item.model, item.colour, item.material, item.size && `size ${item.size}`].filter(Boolean);
  return (
    <motion.li layout="position" transition={move}
      className="grid grid-cols-[2rem_1fr_auto] items-baseline gap-3 border-b border-rule py-3 last:border-0">
      <span className="text-sm text-muted">{rank}</span>
      <div className="min-w-0">
        <p className="truncate font-[family-name:var(--font-tag)] text-lg leading-snug" lang="">{item.title}</p>
        <p className="mt-0.5 truncate text-xs text-muted">
          {facts.length ? facts.join(", ") : "No attributes filled in by the seller"} · {condition(item.condition)} · {item.market}
        </p>
      </div>
      <div className="text-right">
        <p className="font-[family-name:var(--font-tag)] text-lg">{eur(item.price)}</p>
        {moved !== null && moved !== 0 && (
          <p className={`text-xs ${moved > 0 ? "text-gain" : "text-loss"}`}
             aria-label={moved > 0 ? `${moved} places higher than in the other design` : `${-moved} places lower than in the other design`}>
            {moved > 0 ? `${moved} higher` : `${-moved} lower`}
          </p>
        )}
        {moved === null && <p className="text-xs text-garnet">only here</p>}
      </div>
    </motion.li>
  );
}
