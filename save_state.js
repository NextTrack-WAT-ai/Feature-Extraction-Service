// save_state.js
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: false // You can set to true once you're confident
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  // Go to YouTube and log in manually
  await page.goto('https://youtube.com');
  console.log("Log into YouTube in the browser window...");
  await page.waitForTimeout(60000); // Wait 60s for manual login

  // Save authenticated state
  await context.storageState({ path: 'storageState.json' });

  console.log('✅ Saved session state to storageState.json');
  await browser.close();
})();
