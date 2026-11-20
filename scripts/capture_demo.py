"""Exercise the live UI and capture its real interactions for the README."""

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--browser", help="Optional Chrome/Chromium executable")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    frames, durations, errors = [], [], []
    with sync_playwright() as playwright:
        options = {"headless": True}
        if args.browser:
            options["executable_path"] = args.browser
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000}, device_scale_factor=1
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector(".candidate.best")
        assert page.locator("#body-type option").count() == 8
        assert page.locator("#engine option").count() == 2
        assert page.locator(".split-summary > div").first.inner_text().startswith("训练集")
        page.add_style_tag(content="html {scroll-behavior:auto!important}")

        def frame(duration=1500):
            page.wait_for_timeout(150)
            picture = Image.open(BytesIO(page.screenshot())).convert("RGB")
            frames.append(picture.resize((1080, 750), Image.Resampling.LANCZOS))
            durations.append(duration)

        def scroll_to(selector):
            start = page.evaluate("window.scrollY")
            end = page.locator(selector).evaluate(
                "(e) => e.getBoundingClientRect().top + window.scrollY - 30"
            )
            for step in range(1, 6):
                page.evaluate(
                    "(y) => window.scrollTo(0, y)", start + (end - start) * step / 5
                )
                frame(90)

        assert page.locator("#metric-rows").inner_text() == "192,355"
        frame(1800)
        scroll_to("#workspace")
        page.locator("#height").fill("175")
        page.locator("#weight").fill("70")
        page.locator("#body-type").select_option("athletic")
        page.locator("#predict-btn").click()
        page.wait_for_selector(".candidate.best")
        frame(2400)
        page.screenshot(path=str(output / "workspace.png"))

        page.locator("#item-select").select_option(index=1)
        assert page.locator(".candidate").count() == 0
        page.locator("#predict-btn").click()
        page.wait_for_selector(".candidate.best")
        frame(2000)
        scroll_to(".review-panel")
        page.locator("#extract-btn").click()
        page.wait_for_selector(".evidence")
        assert page.locator(".evidence").count() == 3
        assert "This dress runs small." in page.locator("#review-results").inner_text()
        frame(2600)
        page.screenshot(path=str(output / "review-evidence.png"))
        scroll_to("#experiments")
        assert page.locator("#model-table tr").count() == 4
        frame(2400)
        page.screenshot(path=str(output / "experiments.png"))
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(output / "overview.png"), full_page=True)
        frame(1000)

        mobile = browser.new_page(
            viewport={"width": 390, "height": 844}, device_scale_factor=1
        )
        mobile.on("pageerror", lambda error: errors.append(str(error)))
        mobile.goto(args.url, wait_until="networkidle")
        mobile.wait_for_selector(".candidate.best")
        assert mobile.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ), "Mobile overflow"
        mobile.locator("#extract-btn").click()
        mobile.wait_for_selector(".evidence")
        assert mobile.locator(".evidence").count() == 3
        browser.close()
    assert not errors, errors
    frames[0].save(
        output / "workbuddy-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(
        "Desktop + mobile flows passed; screenshots and real UI animation saved to",
        output,
    )


if __name__ == "__main__":
    main()
