"""Verify all nine full-bank sample views on the existing local service."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent/'bank4200_samples_view'
URL = 'http://127.0.0.1:8844/generic-full-2175-lora1500-lineage/p4-v2-bank4200-samples/'


def main():
    checks, errors = [], []
    with sync_playwright() as api:
        browser = api.chromium.launch(headless=True,
            executable_path='/data/gaoya/agent-data/cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
            args=['--disable-gpu', '--disable-gpu-compositing', '--no-proxy-server', '--no-sandbox'])
        page = browser.new_page(viewport=dict(width=1440, height=1000))
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('response', lambda response: errors.append(f'HTTP {response.status}: {response.url}') if response.status >= 400 else None)
        assert page.goto(URL, wait_until='load').status == 200
        page.wait_for_function('typeof data !== "undefined" && !!data')
        for split in ('train', 'val', 'test'):
            page.select_option('#split', split)
            for family in ('barrier', 'door', 'gap'):
                page.click(f'[data-family="{family}"]', force=True)
                page.wait_for_function('document.querySelectorAll(".sample img").length === 4 && Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
                current = page.evaluate('''() => ({keys:Array.from(document.querySelectorAll('.sample h2')).map(x=>x.textContent),
                    error:document.querySelector('#error').textContent,
                    overflow:document.documentElement.scrollWidth>innerWidth})''')
                assert not current['error'] and not current['overflow'], current
                assert all(key.startswith(split+'_'+family+'_') for key in current['keys'])
                checks.append(dict(split=split, family=family, **current))
        page.select_option('#split', 'train')
        page.click('[data-family="gap"]', force=True)
        page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
        page.screenshot(path=str(OUT/'desktop.png'))
        page.set_viewport_size(dict(width=390, height=844))
        page.select_option('#split', 'test')
        page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.screenshot(path=str(OUT/'mobile.png'))
        page.click('summary', force=True)
        assert page.locator('details').get_attribute('open') is not None
        assert not errors, errors
        browser.close()
    (OUT/'browser_checks.json').write_text(json.dumps(dict(groups=checks, errors=errors,
        mobile_no_overflow=True, coverage_toggle=True), indent=2)+'\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
