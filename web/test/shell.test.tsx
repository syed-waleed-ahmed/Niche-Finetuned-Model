import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { STORAGE_KEY } from "../src/state/settings";
import { THEME_STORAGE_KEY } from "../src/state/theme";
import { HEALTH, install, json, resetViewport, setViewport } from "./helpers";

beforeEach(() => {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey: "fa_test" }));
  Element.prototype.scrollIntoView = vi.fn();
  install({ "/health": () => json(HEALTH) });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  resetViewport();
});

describe("theme", () => {
  it("cycles system, light and dark from the header and persists the choice", async () => {
    const user = userEvent.setup();
    render(<App />);
    const root = document.documentElement;
    const toggle = screen.getByRole("button", { name: "Theme: System" });
    expect(root).not.toHaveAttribute("data-theme");

    await user.click(toggle);
    expect(root).toHaveAttribute("data-theme", "light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(toggle).toHaveAccessibleName("Theme: Light");

    await user.click(toggle);
    expect(root).toHaveAttribute("data-theme", "dark");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(screen.getByRole("radio", { name: "Dark" })).toBeChecked();

    await user.click(toggle);
    expect(root).not.toHaveAttribute("data-theme");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("system");
  });

  it("restores a saved theme on load", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "dark");
    render(<App />);
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    expect(screen.getByRole("button", { name: "Theme: Dark" })).toBeInTheDocument();
  });

  it("can be chosen explicitly in Settings", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("radio", { name: "Light" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(screen.getByRole("button", { name: "Theme: Light" })).toBeInTheDocument();
  });

  it("still works when storage throws", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Theme: System" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
  });
});

describe("side panel", () => {
  it("is an inline complementary panel on wide screens", () => {
    setViewport("wide");
    render(<App />);
    expect(screen.getByRole("complementary", { name: "Settings and tools" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button", { name: "Settings" })).not.toHaveAttribute("aria-expanded");
  });

  it("is a drawer on narrow screens: opens, closes on Escape and restores focus", async () => {
    setViewport("narrow");
    const user = userEvent.setup();
    render(<App />);

    const trigger = screen.getByRole("button", { name: "Settings" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByRole("tab", { name: "Settings" })).toBeNull();

    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Settings and tools" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    await waitFor(() => expect(screen.getByRole("button", { name: "Close panel" })).toHaveFocus());
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
    // The rest of the page is out of reach while the drawer is open.
    expect(screen.getByRole("main", { hidden: true })).toHaveAttribute("inert");

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(trigger).toHaveFocus());
    expect(screen.getByRole("main")).not.toHaveAttribute("inert");
  });

  it("closes from the close button and from the backdrop", async () => {
    setViewport("narrow");
    const user = userEvent.setup();
    render(<App />);
    const trigger = screen.getByRole("button", { name: "Settings" });

    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Close panel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(trigger).toHaveFocus());

    await user.click(trigger);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.click(screen.getByTestId("drawer-backdrop"));
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("moves between tabs with the arrow keys", async () => {
    const user = userEvent.setup();
    render(<App />);
    const settingsTab = screen.getByRole("tab", { name: "Settings" });
    settingsTab.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Status" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Status" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Code" })).toHaveFocus();
  });
});
