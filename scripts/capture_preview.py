"""Optional visual smoke check. Requires playwright and its Chromium headless shell."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", default="docs/preview.png")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1600}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.get_by_text("Follow the precision–recall curve", exact=True).wait_for(timeout=60000)
        page.locator("[data-testid='stPlotlyChart']").wait_for(timeout=60000)
        page.locator("[data-testid='stImage'] img").wait_for(timeout=60000)
        page.screenshot(path=args.output, full_page=True)
        if errors:
            raise RuntimeError(errors)
        print(f"Screenshot: {Path(args.output).resolve()}; browser errors: {len(errors)}")
        browser.close()


if __name__ == "__main__":
    main()
