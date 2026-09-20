"""Read-only browser checks for the static comparison report."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    report = Path(__file__).resolve().parent/'two_round_release/reports'
    rows = []
    with sync_playwright() as api:
        browser = api.chromium.launch(headless=True,
            executable_path='/data/gaoya/agent-data/cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
            args=['--disable-gpu', '--disable-gpu-compositing', '--no-sandbox'])
        for width, height in ((1366, 900), (390, 844)):
            page = browser.new_page(viewport=dict(width=width, height=height))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto((report/'index.html').as_uri(), wait_until='load')
            result = page.evaluate('''() => ({images: Array.from(document.images).map(i =>
                ({loaded:i.complete && i.naturalWidth>0, width:i.naturalWidth})),
                overflow:document.documentElement.scrollWidth>innerWidth})''')
            if errors or result['overflow'] or not all(i['loaded'] for i in result['images']):
                raise AssertionError((errors, result))
            page.screenshot(path=str(report/f'page_{width}.png'))
            rows.append(dict(width=width, height=height, errors=errors, **result))
            page.close()
        browser.close()
    (report/'browser_checks.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
