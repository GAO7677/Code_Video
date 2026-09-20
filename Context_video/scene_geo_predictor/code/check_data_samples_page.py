"""Browser checks for sample provenance views and synchronized GT playback."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent/'data_samples_view'
URL = 'http://127.0.0.1:8844/generic-full-2175-lora1500-lineage/p4-v2-data-samples/'


def main():
    results = []
    errors = []
    with sync_playwright() as api:
        browser = api.chromium.launch(headless=True,
            executable_path='/data/gaoya/agent-data/cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
            args=['--disable-gpu', '--disable-gpu-compositing', '--no-proxy-server', '--no-sandbox'])
        page = browser.new_page(viewport=dict(width=1440, height=1000))
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('response', lambda response: errors.append(f'HTTP {response.status}: {response.url}') if response.status >= 400 else None)
        response = page.goto(URL, wait_until='load')
        assert response.status == 200
        print('PAGE', page.title(), page.locator('#kind').count(), flush=True)
        if not page.locator('#kind').count():
            raise AssertionError(page.content()[:1800])
        page.wait_for_function('typeof data !== "undefined" && !!data')
        for kind in ('control', 'train'):
            page.select_option('#kind', kind)
            for family in ('gap', 'door', 'barrier'):
                page.click(f'[data-family="{family}"]')
                page.wait_for_function('cards.length === 4 && cards.every(c => c.images.every(im => im.complete && im.naturalWidth > 0))')
                page.locator('#frame').evaluate("e=>{e.value='23';e.dispatchEvent(new Event('input',{bubbles:true}));}")
                check = page.evaluate('''() => ({cards:cards.length, family:group.family, kind:group.kind,
                    errors:document.querySelector('#error').textContent,
                    overflow:document.documentElement.scrollWidth > innerWidth,
                    nonblank:cards.every(c=>{const p=c.canvas.getContext('2d').getImageData(0,0,1280,720).data;
                        const values=new Set();for(let i=0;i<p.length;i+=4096)values.add(p[i]+256*p[i+1]+65536*p[i+2]);return values.size>30;})})''')
                assert check['nonblank'] and not check['overflow'] and not check['errors'], check
                assert check['family'] == family and check['kind'] == kind
                results.append(check)
        page.select_option('#kind', 'control')
        page.click('[data-family="gap"]')
        page.wait_for_function('cards.length === 4 && cards.every(c => c.images.every(im => im.complete && im.naturalWidth > 0))')
        page.locator('#frame').evaluate("e=>{e.value='7';e.dispatchEvent(new Event('input',{bubbles:true}));}")
        page.click('#play')
        page.wait_for_timeout(500)
        assert page.locator('#frame').input_value() != '7'
        page.click('#play')
        page.locator('#overlay').uncheck()
        page.locator('#overlay').check()
        page.locator('#future').uncheck()
        page.locator('#future').check()
        page.locator('#frame').evaluate("e=>{e.value='23';e.dispatchEvent(new Event('input',{bubbles:true}));}")
        page.screenshot(path=str(OUT/'desktop.png'))
        page.set_viewport_size(dict(width=390, height=844))
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(OUT/'mobile.png'))
        assert not errors, errors
        browser.close()
    (OUT/'browser_checks.json').write_text(json.dumps(dict(groups=results, errors=errors,
        playback_advanced=True, toggles_tested=True, mobile_no_overflow=True), indent=2)+'\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
