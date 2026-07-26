import { chromium } from "playwright";
const log=(...a)=>console.log(...a);
const browser = await chromium.launch();
const ctx = await browser.newContext({viewport:{width:1440,height:1100}});
const page = await ctx.newPage();
page.on("pageerror",e=>log("[pageerror]",e.message));
await page.goto("http://localhost:5173/",{waitUntil:"domcontentloaded"});
await page.getByRole("button",{name:"Geometry"}).first().click();
await page.waitForSelector('[data-testid="geo-view"][data-ready="1"]',{timeout:180000});
await page.waitForTimeout(3000);
const grab = () => page.evaluate(()=>{
  const gl=document.querySelector('[data-testid="geo-canvas"] canvas');
  const c=document.createElement('canvas');c.width=gl.width;c.height=gl.height;
  const x=c.getContext('2d');x.drawImage(gl,0,0);
  return Array.from(x.getImageData(0,0,c.width,c.height).data);
});
const diff=(a,b)=>{let n=0,max=0;for(let i=0;i<a.length;i+=4){const d=Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2]);if(d>3){n++;max=Math.max(max,d);}}return {px:n,max};};
const box=await page.locator('[data-testid="geo-canvas"] canvas').boundingBox();
const cx=box.x+box.width/2, cy=box.y+box.height/2;

log("--- baseline (no interaction) ---");
let a=await grab(); await page.waitForTimeout(2000); let b=await grab();
log("idle 2s diff:",JSON.stringify(diff(a,b)));

log("--- after drag ---");
await page.mouse.move(cx,cy);await page.mouse.down();
await page.mouse.move(cx+90,cy+20,{steps:10});await page.mouse.up();
for (const wait of [1000,3000,6000,10000]) {
  await page.waitForTimeout(wait===1000?1000:wait-(wait===3000?1000:wait===6000?3000:6000));
  const p=await grab(); await page.waitForTimeout(1500); const q=await grab();
  log(`t=${wait}ms after drag, diff over 1.5s:`,JSON.stringify(diff(p,q)));
}
log("aria after drag:",await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));

log("--- spin ON then drag ---");
await page.locator('[data-testid="geo-autorotate"]').click();
await page.waitForTimeout(500);
let s1=await grab(); await page.waitForTimeout(1500); let s2=await grab();
log("spin on, diff 1.5s:",JSON.stringify(diff(s1,s2)));
await page.mouse.move(cx,cy);await page.mouse.down();await page.mouse.move(cx+40,cy,{steps:6});await page.mouse.up();
await page.waitForTimeout(6000);
let t1=await grab(); await page.waitForTimeout(2000); let t2=await grab();
log("after drag-stops-spin, diff 2s:",JSON.stringify(diff(t1,t2)),"aria:",await page.locator('[data-testid="geo-autorotate"]').getAttribute("aria-pressed"));
await browser.close();
