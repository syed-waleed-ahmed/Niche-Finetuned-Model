import { fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { MessageList } from "../src/components/MessageList";
import type { Turn } from "../src/state/conversation";

// jsdom does no layout: give every element a fixed scrollable height and a writable
// scrollTop so the scroll-follow logic can be observed.
const SCROLL_HEIGHT = 900;
const CLIENT_HEIGHT = 300;
const tops = new WeakMap<Element, number>();

beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
    configurable: true,
    get: () => SCROLL_HEIGHT,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => CLIENT_HEIGHT,
  });
  Object.defineProperty(HTMLElement.prototype, "scrollTop", {
    configurable: true,
    get(this: Element) {
      return tops.get(this) ?? 0;
    },
    set(this: Element, value: number) {
      tops.set(this, Math.max(0, Math.min(value, SCROLL_HEIGHT - CLIENT_HEIGHT)));
    },
  });
});

afterEach(() => {
  for (const prop of ["scrollHeight", "clientHeight", "scrollTop"]) {
    Reflect.deleteProperty(HTMLElement.prototype, prop);
  }
});

const user = (id: string): Turn => ({ id, role: "user", content: `question ${id}` });
const rate = async () => null;
const BOTTOM = SCROLL_HEIGHT - CLIENT_HEIGHT;

function scroller(container: HTMLElement): HTMLElement {
  const el = container.querySelector<HTMLElement>(".messages");
  if (!el) throw new Error("no .messages element");
  return el;
}

describe("MessageList scrolling", () => {
  it("shows the welcome screen from the top", () => {
    const { container } = render(<MessageList turns={[]} onRate={rate} welcome={<p>Welcome</p>} />);
    expect(scroller(container).scrollTop).toBe(0);
  });

  it("follows the first answer even if the reader scrolled the welcome screen", () => {
    const { container, rerender } = render(
      <MessageList turns={[]} onRate={rate} welcome={<p>Welcome</p>} />,
    );
    const el = scroller(container);
    el.scrollTop = 120;
    fireEvent.scroll(el);

    rerender(<MessageList turns={[user("1")]} onRate={rate} welcome={<p>Welcome</p>} />);
    expect(el.scrollTop).toBe(BOTTOM);
  });

  it("stops following once the reader scrolls up during a conversation", () => {
    const { container, rerender } = render(
      <MessageList turns={[user("1")]} onRate={rate} welcome={<p>Welcome</p>} />,
    );
    const el = scroller(container);
    expect(el.scrollTop).toBe(BOTTOM);

    el.scrollTop = 100;
    fireEvent.scroll(el);
    rerender(<MessageList turns={[user("1"), user("2")]} onRate={rate} welcome={<p>Welcome</p>} />);
    expect(el.scrollTop).toBe(100);
  });

  it("returns to the top when the conversation is cleared", () => {
    const { container, rerender } = render(
      <MessageList turns={[user("1")]} onRate={rate} welcome={<p>Welcome</p>} />,
    );
    const el = scroller(container);
    expect(el.scrollTop).toBe(BOTTOM);

    rerender(<MessageList turns={[]} onRate={rate} welcome={<p>Welcome</p>} />);
    expect(el.scrollTop).toBe(0);
  });
});
