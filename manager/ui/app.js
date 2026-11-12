const {invoke} = window.__TAURI__.core;
const {listen} = window.__TAURI__.event;
const OFFICIAL_REPOSITORY="https://github.com/deepbeepmeep/WanGP";
const state = {
  page:"overview", preferences:{installations:[],selected:null,port:7860}, jobs:new Map(), pending:new Map(),
  hardware:null, catalog:null, inspection:null, versions:[], preview:null, demo:null, message:null, runtime_error:null,
  form:{source:"",root:"",release_id:"",profile:"",device:"",extras:true,environment:"auto",operation:"install"},
  registration:{root:"",python:""}, detected:{environments:[],loading:false,error:null}, managerUpdate:null, latest:null, selectedLog:null, consoleExpanded:false, progressExpanded:false, launchDraft:null, preferencesDraft:null, authenticationDraft:{password:""}, savingPreferences:false,
  conda:{environments:[],warnings:[],loading:false,loaded:false,error:null},
  renderedView:null, scrollPositions:new Map(), flowKey:"install", flows:new Map()
};
state.remote={manual:false,kind:"github",repository:OFFICIAL_REPOSITORY,commit:"",choices:[],page:1,has_more:false,catalog_id:null,loaded_repository:null};
const initialForm={...state.form},initialRemote={...state.remote};
state.flows.set("install",{form:state.form,remote:state.remote,catalog:null,preview:null});
for(const key of ["form","remote","catalog","preview"])Object.defineProperty(state,key,{get(){return state.flows.get(state.flowKey)[key];},set(value){state.flows.get(state.flowKey)[key]=value;}});
function selectFlow(page){
  if(!["install","maintenance"].includes(page))return;
  state.flowKey=page==="install"?"install":"maintenance:"+(selected()?.root||"");
  if(!state.flows.has(state.flowKey)){
    const flow={form:{...initialForm,root:selected()?.root||"",operation:"update"},remote:{...initialRemote,choices:[]},catalog:null,preview:null};
    if(selected()?.test_workspace){Object.assign(flow.remote,{manual:true,kind:"local"});flow.form.source=selected().demo_releases?.[0]?.path||"";flow.form.profile="Demo";}
    state.flows.set(state.flowKey,flow);
  }
}
const currentFlow=()=>state.flows.get(state.flowKey);
const viewKey=()=>state.page+(state.page==="install"&&state.registerView?":registration":"");
const applyLabel=plan=>plan.operation==="install"?(state.catalog?.profiles.Demo?"Install test installation":"Install WanGP"):plan.operation==="rollback"?"Restore this version":plan.operation==="update"?"Update WanGP":"Apply changes";
const icons = {
  home:'<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M9 21v-8h6v8"/>',
  install:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  maintain:'<path d="M20 7a6 6 0 0 1-8 8l-6 6-3-3 6-6a6 6 0 0 1 8-8l-4 4 3 3z"/>',
  diagnostic:'<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  activity:'<path d="M5 4h14v17H5zM9 9h6M9 13h6M9 17h3"/>',
  settings:'<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/>',
  box:'<path d="m3 7 9-4 9 4-9 5zM3 7v10l9 4 9-4V7M12 12v9M7 5l10 5"/>',
  play:'<path d="m8 4 13 8-13 8z"/>',
  copy:'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  save:'<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h12l4 4v12a2 2 0 0 1-2 2Z"/><path d="M7 3v6h10V3M7 21v-8h10v8"/>',
  minimize:'<path d="M5 12h14"/>',
  expand:'<path d="M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5"/>',
  collapse:'<path d="M3 8h5V3M21 8h-5V3M8 21v-5H3M16 21v-5h5"/>'
};
const esc = value => String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
// Templates contain only our markup; all worker/user text is escaped at its insertion point.
const html = (template,values={}) => template.replace(/\{(\w+)\}/g,(_,key)=>values[key]??"");
const svg = name => '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+(icons[name]||icons.box)+'</svg>';
const pretty = value => esc(JSON.stringify(value,null,2));
const selected = () => state.preferences.installations.find(x=>x.root===state.preferences.selected);
const usesConda = item => item?.maintenance==="conda"||item?.ownership==="conda"||Boolean(item?.inspection?.conda_environment);
const canMaintain = item => item?.ownership==="managed"||["conda","venv"].includes(item?.maintenance)||["conda","venv"].includes(item?.ownership);
const server = () => [...state.jobs.values()].find(x=>x.action==="launch"&&!x.done&&x.args?.root===selected()?.root);
// The official builds newer than an installation's: after its own build when the newest builds list it, else those of a higher version.
function updatesFor(item){
  const builds=state.latest?.versions||[],release=String(item?.release||"");
  const version=release.match(/^\d+(?:\.\d+)*/)?.[0],commit=release.match(/([0-9a-f]{7,40})$/)?.[1],index=commit?builds.findIndex(x=>x.commit.startsWith(commit)):-1;
  if(!builds.length||item?.test_workspace||!version&&!commit)return null;  // a checkout of unknown version: nothing to say
  const newer=index>=0?builds.slice(0,index):version?builds.filter(x=>newerVersion(x.version,version)):[];
  return {latest:builds[0],newer,more:index<0&&newer.length===builds.length&&state.latest.has_more};
}
function whatsNew(item){
  const news=updatesFor(item);
  if(!news?.newer.length)return "";
  return html('<section class="card whats-new"><div class="row"><div><h3>{title}</h3><p class="hint">What changed since your version, newest first.</p></div>{update}</div><ul class="changes">{items}</ul>{more}</section>',{
    title:"WanGP "+esc(news.latest.version)+" is available",
    update:canMaintain(item)?button("navigate","Update WanGP","primary",!!server()||!!busyJob(),'data-page="maintenance"'):'<span class="hint">Update this checkout with Git.</span>',
    items:news.newer.slice(0,12).map(x=>'<li><strong>'+esc(x.version)+'</strong><span class="hint">'+esc(new Date(x.date).toLocaleDateString())+'</span><span>'+esc((x.message||"").split("\n")[0])+'</span></li>').join(""),
    more:news.newer.length>12||news.more?'<p class="hint">And older changes: the full list is in Update WanGP.</p>':""
  });
}
function checkUpdates(){  // once per start, quietly: offline, Manager shows no notice; without an installation there is nothing to compare
  if(selected())state.latestCheck=run("latest_versions").then(value=>{state.latest=value;render();}).catch(()=>{});
  run("manager_update").then(value=>{state.managerUpdate=value;render();}).catch(()=>{});
}
const busyJob = () => [...state.jobs.values()].find(x=>!x.done&&["preview","apply","recover","remove_version","demo","register","sources","catalog"].includes(x.action));
const launchLog = () => [...state.jobs.values()].reverse().find(x=>x.action==="launch"&&x.args?.root===selected()?.root);
const usingRemote = () => !state.remote.manual||state.remote.kind==="github";
const sourceArgs = () => usingRemote()?{repository:state.remote.repository,commit:state.remote.commit}:{source:state.form.source,release_id:state.form.release_id};
const disabled = value => value?"disabled":"";
const option = (value,label,current) => html('<option value="{value}" {selected}>{label}</option>',{value:esc(value),label:esc(label),selected:value===current?"selected":""});
const button = (action,label,style="",off=false,attributes="") => html('<button data-action="{action}" class="{style}" {off} {attributes}>{label}</button>',{action,style,off:disabled(off),attributes,label});
const iconButton = (action,icon,label,off=false,attributes="") => button(action,svg(icon),"icon-button",off,'title="'+esc(label)+'" aria-label="'+esc(label)+'" '+attributes);
const heading = (label,title,description,extra="") => html('<div class="page-head"><div><div class="eyebrow">{label}</div><h1>{title}</h1><p>{description}</p></div>{extra}</div>',{label,title,description,extra});
const notice = (message,type="") => '<div class="notice '+type+'">'+message+'</div>';
const field = (group,key,label,placeholder="",kind=null) => html('<div class="field"><label for="{id}">{label}</label><div class="input-action"><input id="{id}" data-group="{group}" data-key="{key}" value="{value}" placeholder="{placeholder}" autocomplete="off">{browse}</div></div>',{
  id:group+"-"+key,group,key,label,value:esc(state[group][key]),placeholder:esc(placeholder),
  browse:kind?button("browse","Browse…","",false,'data-group="'+group+'" data-key="'+key+'" data-kind="'+kind+'"'):""
});
const selectField = (key,label,choices,current) => html('<div class="field"><label for="form-{key}">{label}</label><select id="form-{key}" data-group="form" data-key="{key}">{choices}</select></div>',{key,label,choices:choices.map(x=>option(x[0],x[1],current)).join("")});
function progressMarkup(){
  const job=busyJob();
  if(!job){
    const plan=state.preview?.plan;
    return plan&&["install","maintenance"].includes(state.page)&&!state.registerView?'<section id="progress-area" class="operation-dock ready" aria-label="Installation ready"><div class="row"><div><strong>Ready · '+esc(plan.target_release_id)+'</strong><p class="path">'+esc(plan.installation_root)+'</p><p>'+(plan.reuse_environment?'Reuse the compatible environment.':'Create a new environment.')+' '+plan.changes.length+' source file change'+(plan.changes.length===1?'':'s')+'.</p></div>'+button('apply',applyLabel(plan),'primary',server()||state.runtime_error)+'</div></section>':'<div id="progress-area"></div>';
  }
  const progress=operationProgress(job);
  return '<section id="progress-area" class="operation-dock" aria-label="Installation progress"><div class="row"><div class="operation-heading" role="status"><span class="spinner"></span><strong id="progress-title">'+esc(progress.title)+'</strong></div><div class="operation-actions">'+button('toggle-progress',state.progressExpanded?'Hide output':'Show output','small')+(job.action==="remove_version"?"":button('cancel','Cancel','quiet small',false,'data-id="'+job.id+'"'))+'</div></div><p id="progress-message">'+esc(progress.message)+'</p><div id="operation-meter" class="operation-meter '+(progress.total?'determinate':'')+'"><progress id="operation-progress" aria-label="Current step progress" '+(progress.total?'max="'+progress.total+'" value="'+progress.completed+'"':'')+'></progress><div class="progress" role="progressbar" aria-label="Current step in progress"></div></div><p id="progress-output" class="operation-output">'+esc(progress.output)+'</p>'+(state.progressExpanded?'<pre id="operation-log" class="log" tabindex="0" aria-label="Installation output">'+esc(jobLog(job))+'</pre>':'')+'</section>';
}
function operationProgress(job){
  const last=[...job.events].reverse().find(e=>e.title)||{};
  const line=[...job.events].reverse().find(e=>e.type==="log"&&e.text.trim());
  return {title:last.title||({preview:"Preparing Installation",apply:"Installing WanGP",remove_version:"Removing Saved Version",recover:"Recovering Installation",demo:"Preparing Test Installation",register:"Reading Installation",sources:"Finding WanGP Versions",catalog:"Reading Installation Configuration"}[job.action]),
    message:last.message||({building_environment:"Creating Python and downloading the required packages.",validating_environment:"Checking installed packages and GPU compatibility.",caching_release:"Saving source files for installation and rollback.",backing_up:"Saving the previous version before applying changes.",activating:"Activating the prepared version."}[last.phase])||"Working… Live output appears below as it becomes available.",
    total:Number.isFinite(last.total)&&last.total>0?last.total:null,completed:Math.max(0,last.completed||0),
    output:line?line.text.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g,"").split(/[\r\n]+/).filter(x=>x.trim()).at(-1)?.slice(-220)||"":""};
}
function updateOperationProgress(){
  const job=busyJob();if(!job)return;
  const current=operationProgress(job);
  for(const [id,text] of [["progress-title",current.title],["progress-message",current.message],["progress-output",current.output]]){const node=document.getElementById(id);if(node)node.textContent=text;}
  document.getElementById("operation-meter")?.classList.toggle("determinate",!!current.total);
  const bar=document.getElementById("operation-progress");
  if(bar){if(current.total){bar.max=current.total;bar.value=current.completed;}else bar.removeAttribute("value");}
  const log=document.getElementById("operation-log");
  if(log){const bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-30;log.textContent=jobLog(job);if(bottom)log.scrollTop=log.scrollHeight;}
}
function render(){
  const focused=document.activeElement?.id,caret=document.activeElement?.selectionStart;
  const scroll=document.scrollingElement;
  if(state.renderedView)state.scrollPositions.set(state.renderedView,scroll?.scrollTop||0);
  const logPositions=["server-log","live-log","operation-log"].map(id=>{const node=document.getElementById(id);return node?{id,top:node.scrollTop,bottom:node.scrollTop+node.clientHeight>=node.scrollHeight-30}:null;}).filter(Boolean);
  if(!state.preferences.installations.length){state.page="install";selectFlow("install");}
  const nextView=viewKey(),sameView=state.renderedView===nextView;
  const scrollTop=state.scrollPositions.get(nextView)||0;
  state.renderedView=nextView;
  const nav=state.preferences.installations.length?[["overview","home","Overview"],["install","install","Install / Add"],["maintenance","maintain","Update WanGP"],["diagnostics","diagnostic","Diagnostics"],["activity","activity","Activity"],["preferences","settings","Preferences"]]:[["install","install","Install / Add"]];
  document.querySelector("#app").innerHTML=html(
    '<div class="shell" {shellState}><aside class="sidebar"><div class="brand"><img class="brand-icon" src="manager.png" alt=""><div><div class="brand-name">WanGP</div><div class="brand-sub">Manager</div></div></div><div class="nav-label">YOUR INSTALLATIONS</div><nav class="nav">{nav}</nav><div class="sidebar-bottom"><div><span class="dot {warn}"></span>{runtime}</div><div class="badge-label">VERSION {version} · EXPERIMENTAL</div>{managerNew}{star}</div></aside><main class="main"><header class="topbar"><div class="breadcrumb">WanGP Manager / <strong>{page}</strong></div>{picker}</header><div class="content">{error}{message}{progress}{content}</div></main></div>{expandedConsole}<dialog id="close-dialog"><h2>Work is still running</h2><p>Keep Manager open, or stop its servers and cancel active operations before exiting. Interrupted installation work can be recovered when you return.</p><div class="actions">{keep}{exit}</div></dialog>',{
      shellState:state.consoleExpanded?'inert aria-hidden="true"':"",
      expandedConsole:state.consoleExpanded?consoleMarkup(launchLog(),"server-log","Start WanGP to see its console output here."):"",
      nav:nav.map(([page,icon,label])=>button("navigate",svg(icon)+label,state.page===page?"active":"",false,'data-page="'+page+'"')).join(""),
      warn:state.runtime_error?"warn":"",runtime:state.runtime_error?"Runtime needs attention":"Local runtime ready",version:esc(state.version||"0.1.0"),
      managerNew:state.managerUpdate&&newerVersion(state.managerUpdate.version,state.version||"0")?button("navigate","Manager "+esc(state.managerUpdate.version)+" is available","text-link small manager-new",false,'data-page="preferences"'):"",
      star:button("github-star","★ Star WanGP on GitHub","text-link small manager-new",false,'title="Opens the WanGP project page in your browser"'),
      page:nav.find(x=>x[0]===state.page)?.[2],
      picker:state.preferences.installations.length?'<select id="installation-picker" aria-label="Active installation">'+state.preferences.installations.map(x=>option(x.root,(x.test_workspace?"Test installation":x.name)+(x.test_workspace?" · Test":""),selected()?.root)).join("")+'</select>':'<span class="pill">WanGP installations</span>',
      error:state.runtime_error?notice(esc(state.runtime_error),"error"):"",
      message:state.message?'<div class="toast notice '+state.message.type+'" role="'+(state.message.type==="error"?'alert':'status')+'"><div class="row"><span>'+esc(state.message.text)+'</span>'+button("dismiss","×","quiet small",false,'aria-label="Dismiss notification"')+'</div></div>':"",
      progress:progressMarkup(),content:({overview,install,maintenance,diagnostics,activity,preferences}[state.page])(),
      keep:button("keep-open","Keep working"),exit:button("exit","Stop and exit","danger")
    });
  if(sameView&&focused){const input=document.getElementById(focused);if(input){input.focus({preventScroll:true});if(typeof caret==="number"&&input.setSelectionRange)try{input.setSelectionRange(caret,caret);}catch{}}}
  for(const position of logPositions){const node=document.getElementById(position.id);if(node)node.scrollTop=position.bottom?node.scrollHeight:position.top;}
  if(scroll)scroll.scrollTop=scrollTop;
}
function overview(){
  const item=selected(),live=server();
  if(!item)return heading("WELCOME","Make room for creating.","Install WanGP, keep a known-good version, and get back to your work.")+html(
    '<section class="card hero"><div class="hero-header"><div><div class="pill purple">A home for your installations</div><div class="spacer"></div><h2>Your first installation starts here.</h2><p class="subtitle">Bring an existing WanGP installation into Manager, or set up a separate one. Your models, settings and outputs stay with you.</p><div class="actions">{setup}{register}</div></div><div class="hero-art">{box}</div></div></section><div class="section-caption">GET TO KNOW MANAGER</div><div class="grid2"><section class="card"><div class="icon-tile">{maintain}</div><h3>A way back when you need it</h3><p class="hint">Preview changes, try an older version and return to a saved checkpoint.</p></section><section class="card"><div class="icon-tile">{play}</div><h3>Try a small test installation</h3><p class="hint">Explore installation, launching and version switching with a tiny sample. No models or GPU downloads.</p>{demo}</section></div><p class="footer-note">This local preview uses source folders and prepared release bundles. Online release browsing will arrive with verified distribution.</p>',{
      setup:button("navigate","Set up WanGP","primary",false,'data-page="install"'),register:button("register-view","Add existing installation"),box:svg("box"),maintain:svg("maintain"),play:svg("play"),demo:button("demo","Prepare test installation →","text-link",busyJob()||state.runtime_error)
    });
  const status=item.pending_operation?"Recovery needed":live?.stopping?"Stopping":live?.server_url?"Server ready":live?"Starting":"Stopped";
  return heading("YOUR INSTALLATIONS",item.test_workspace?"Your test installation.":"Ready when you are.","Launch WanGP here, or open it in your browser.")+html(
    '<section class="card hero"><div class="hero-header"><div><div class="row"><span class="pill {statusClass}">{status}</span><span class="pill purple">{ownership}</span></div><div class="spacer"></div><h2>{name}</h2><p class="path">{root}</p></div><div class="hero-art">{box}</div></div><div class="actions">{actions}</div>{launching}{url}{console}</section><div class="cards">{cards}</div>{whatsNew}{test}<section class="card"><div class="row"><div><h3>{heading}</h3><p class="hint">{description}</p></div>{manage}</div></section>',{
      whatsNew:whatsNew(item),
      statusClass:live?.server_url?"green":item.pending_operation?"amber":"",status,ownership:item.test_workspace?"Test installation":item.ownership==="managed"?"Managed installation":"Existing installation",name:esc(item.test_workspace?"Test installation":item.name),root:esc(item.root),box:svg("box"),
      actions:button("start",live&&!live.server_url?'<span class="spinner"></span> '+(live.stopping?"Stopping…":"Starting…"):"Start WanGP","primary",live||item.pending_operation||busyJob()||state.runtime_error)+button("navigate","Update WanGP","",live||busyJob()||!canMaintain(item),'data-page="maintenance"')+button("open-app","Open in App","",!live?.server_url)+button("open-browser","Open in Browser","",!live?.server_url)+(live?'<span class="separator"></span>'+button("stop",live.stopping?"Stopping…":"Stop","danger",live.stopping):"")+(item.pending_operation?button("recover","Recover installation","primary"):""),
      launching:live&&!live.server_url?'<div class="launch-progress" role="status"><p class="hint">'+(live.stopping?"Waiting for WanGP to stop…":"Starting WanGP. The opening buttons become available when the server is ready.")+'</p><div class="progress"></div></div>':"",
      console:state.consoleExpanded?"":consoleMarkup(launchLog(),"server-log","Start WanGP to see its console output here."),
      url:live?.server_url?'<p class="footer-note">'+esc(live.server_url)+' · Same server in both views</p>':"",
      cards:[
        ["WanGP version",item.release||"Not activated",(canMaintain(item)?"Source and environment can be restored together":"Source remains under your control")+(updatesFor(item)?"\n"+(updatesFor(item).newer.length?"Update available: WanGP "+updatesFor(item).latest.version:"Up to date"):"")],
        ["Environment",item.ownership==="managed"?"WanGP · uv":usesConda(item)?"Conda":"Registered Python",environmentDetails(item.inspection||state.inspection?.inspection)],
        ["Launch port",state.preferences.port,"Change it in Preferences"]
      ].map(([label,value,detail])=>html('<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',{label,value:esc(value),detail:esc(detail)})).join(""),
      test:item.test_workspace?notice("This test installation runs a small sample web page. It does not contain WanGP models or generation features."):"",
      heading:canMaintain(item)?"Keep a version that works for you.":"Your existing installation stays yours.",
      description:canMaintain(item)&&item.ownership!=="managed"?"Update WanGP and its dependencies, or restore a saved source and environment pair.":item.ownership==="managed"?"Update this installation, try another release or restore a saved version.":"Register a Git checkout with a Conda or virtual environment to enable updates and rollback.",
      manage:button("navigate",canMaintain(item)?"Update WanGP":"Inspect installation","primary",false,'data-page="'+(canMaintain(item)?"maintenance":"diagnostics")+'"')
    });
}
function install(){
  if(state.registerView)return heading("ADD EXISTING","Bring your installation.","Register its folder and Python environment without changing packages.")+
    '<section class="card">'+field("registration","root","WanGP folder","Folder containing wgp.py, or a managed installation root","folder")+button("detect-installation",state.detected.loading?"Finding environment…":"Detect environment","small",!state.registration.root||state.detected.loading)+localEnvironmentField()+'<details><summary>Choose another Conda environment or Python</summary>'+condaEnvironmentField()+field("registration","python","Python executable","Select the Python used by this installation","python")+'</details>'+
    '<div class="form-footer"><p class="hint">Your models, plugins and settings stay in place. Once added, choose Start WanGP or Update WanGP on the Overview page.</p>'+button("register","Add installation","primary",!state.registration.root||busyJob()||state.detected.loading)+'</div></section>'+button("new-view","← Set up a new installation","text-link");
  return heading("SET UP","Install or add WanGP.","Use an existing installation, or let Manager create a new one.")+'<section class="card existing-entry"><div><h2>Already have WanGP?</h2><p class="hint">Use your current installation, including Conda, uv, venv and Desktop setups. Keep your models and settings.</p></div>'+button("register-view","Use existing installation","primary",busyJob())+'</section>'+installForm(false)+review()+
    '<section class="installation-alternatives"><h2>Other ways to get started</h2><div><h3>Already have WanGP?</h3><p class="hint">Add its folder and Python environment to Manager.</p>'+button("register-view","Add existing installation","small")+'</div><details><summary>Try a lightweight test installation</summary><p class="hint">Creates a small sample web page to practice installation, launching and rollback. It does not install WanGP or download models or GPU packages.</p>'+button("demo","Prepare test installation","small",busyJob()||state.runtime_error)+'</details></section>';
}
function localEnvironmentField(){
  const detected=state.detected,current=state.registration.python;
  return '<div class="field detected-environment">'+(detected.error?notice(esc(detected.error),"error"):'')+(detected.environments.length?'<label for="local-environment">Environment found in this installation</label><select id="local-environment">'+option('','Choose an environment',current)+detected.environments.map(x=>option(x.interpreter,x.name+' · Python '+x.python+' · PyTorch '+(x.packages?.torch||'not installed')+(x.saved?' · Saved selection':''),current)).join('')+'</select><p class="hint">'+esc(current||'More than one environment may be available; choose the one you normally use.')+'</p>':state.registration.root&&!detected.loading?'<p class="hint">Choose Detect environment, or select another environment below.</p>':'')+'</div>';
}
function profileName(key,label){return /^(RTX|GTX)_\d{2}$/.test(key)?key.replace("_"," ")+"XX":label;}
function condaEnvironmentField(){
  const {environments,loading,loaded,error,warnings}=state.conda,current=state.registration.python;
  const key=value=>state.hardware?.platform==="win"?value.replace(/\\/g,"/").toLowerCase():value;
  const selectedEnvironment=environments.find(item=>key(item.python)===key(current));
  const choices=[option("","Use the installation's saved environment",current)];
  for(const item of environments)choices.push(option(item.python,item.name+" — "+item.prefix,selectedEnvironment?.python||current));
  if(current&&!selectedEnvironment)choices.push(option(current,"Manually selected Python",current));
  return '<div class="field"><div class="row"><label for="conda-environment">Conda environment</label>'+button("refresh-conda","Refresh","small",loading||state.runtime_error)+'</div><select id="conda-environment" '+disabled(loading)+'>'+choices.join("")+'</select><p class="hint" role="status">'+(loading?'<span class="spinner"></span> Finding Conda environments…':loaded?(environments.length?environments.length+" Conda environment"+(environments.length===1?"":"s")+" found. Select the one you use for this WanGP folder.":"No Conda Python environments found. You can browse to Python below."):"Choose an existing environment or browse to Python below.")+'</p><p class="hint">Discovery and registration do not install or change packages.</p>'+(error?notice(esc(error)+" You can still select Python manually.","error"):"")+warnings.map(message=>notice(esc(message))).join("")+'</div>';
}
function installForm(maintaining){
  const {form,remote}=state,profiles=state.catalog?.profiles||{},devices=state.hardware?.devices||[],online=usingRemote();
  const ready=state.catalog&&form.profile&&(online?remote.commit:form.source);
  const profile=profiles[form.profile];
  return '<fieldset class="install-fields" '+disabled(busyJob())+'><div class="installer"><section class="install-step" aria-labelledby="source-step"><div class="form-title"><span class="step">1</span><div><h3 id="source-step">Choose your WanGP version</h3><p class="hint">'+(remote.manual?'Choose a fork or a local source folder.':'The official repository is selected for you.')+'</p></div></div>'+
    '<div class="source-heading"><label for="repository">Repository</label><label class="check"><input type="checkbox" data-group="remote" data-key="manual" '+(remote.manual?'checked':'')+'>Manual source (forks / local folders)</label></div>'+
    (remote.manual?'<div class="field"><select aria-label="Source type" data-group="remote" data-key="kind">'+option('github','Public GitHub repository',remote.kind)+option('local','Local source or release folder',remote.kind)+'</select></div>':'')+
    (online?'<div class="input-action"><input id="repository" aria-label="Repository" data-group="remote" data-key="repository" value="'+esc(remote.repository)+'" '+(!remote.manual?'readonly':'')+'>'+button('sources','Load versions','',busyJob())+'</div><div class="spacer"></div><div class="field"><label for="source-version">Version to download</label><select id="source-version" data-group="remote" data-key="commit" '+disabled(busyJob()||!remote.choices.length)+'>'+(!remote.choices.length?'<option>Load available versions from GitHub</option>':remote.choices.map(x=>option(x.commit,x.label,remote.commit)).join(''))+'</select><p class="hint">'+(remote.choices.length?'20 builds per page, starting at WanGP 18.888. Each choice installs the exact revision shown.':'Choose Latest for the current version, or an older version to investigate a regression.')+'</p>'+(remote.choices.find(x=>x.commit===remote.commit)?.message?'<p class="commit-message">'+esc(remote.choices.find(x=>x.commit===remote.commit).message)+'</p>':'')+(remote.stale?'<p class="hint">Showing cached build history; the published catalog is unavailable.</p>':remote.origin==='bundled'?'<p class="hint">Showing the build history bundled with this preview.</p>':'')+'</div>'+(remote.has_more?button('older-sources','Show older versions','small',busyJob()):''):
      field('form','source','Source folder','Choose a WanGP checkout or release bundle','folder')+'<div class="row">'+button('catalog','Read source configuration','small',!form.source||busyJob())+'<span class="hint">'+esc(state.catalog?.release||'')+'</span></div>'+(!state.catalog?.bundled?field('form','release_id','Version label','For example, local-preview'):'')+
      (selected()?.test_workspace&&maintaining?'<div class="chips">'+selected().demo_releases.map(x=>button('demo-version','Test '+esc(x.version),'small',busyJob(),'data-source="'+esc(x.path)+'"')).join('')+'</div>':''))+
    '</section><section class="install-step" aria-labelledby="hardware-step"><div class="form-title"><span class="step">2</span><div><h3 id="hardware-step">Confirm your hardware</h3><p class="hint">Only GPUs detected on this computer are listed.</p></div></div>'+
    '<div class="grid2">'+selectField('device','GPU',[["",state.catalog?.profiles.Demo?'Sample installation — no GPU needed':state.hardware?'Choose a detected GPU':'Detecting GPUs…'],...devices.map(x=>[x.id,x.name+(x.memory_bytes?' · '+Math.round(x.memory_bytes/1024**3)+' GB':'' )])],form.device)+
    '<div class="field"><label>Installation profile · Automatic</label><div class="inferred-profile"><strong>'+esc(profile?profileName(form.profile,profile.label):(state.catalog?'No compatible profile':'Waiting for version configuration'))+'</strong></div></div></div>'+
    '<p class="profile-stack">'+esc(profile?'Python '+profile.python+' · '+profile.torch:'Select a version to find the right packages for your GPU.')+'</p>'+
    (state.catalog&&!form.profile?notice(devices.length?'This version has no supported profile for the selected GPU. Choose another detected GPU or version.':'No supported GPU was detected. Check the GPU driver, then refresh hardware.','error'):'')+
    button('refresh-hardware','Refresh hardware','small',busyJob())+
    '</section><section class="install-step" aria-labelledby="destination-step"><div class="form-title"><span class="step">3</span><div><h3 id="destination-step">'+(maintaining?'Update options':'Choose where to install')+'</h3><p class="hint">'+(maintaining?'Keep the current environment when compatible, or create a separate one.':'Python and packages will be installed automatically in this folder.')+'</p></div></div>'+
    (!maintaining?field('form','root','Installation folder','Choose a new or empty folder','folder'):
      '<div class="grid2">'+selectField('operation','Action',[["switch-version","Switch version / downgrade"],["update","Update"],...(selected()?.ownership==="managed"?[["repair","Repair missing files and rebuild"]]:[])],form.operation)+selectField('environment','Environment',[["auto","Automatic"],["current","Reuse current (if compatible)"],["new","Create a new environment"]],form.environment)+'</div>')+
    '<label class="check recommended-components"><input type="checkbox" data-group="form" data-key="extras" '+(form.extras?'checked':'')+'>Include recommended optional components</label>'+
    '<div class="form-footer"><p class="hint">'+(state.preview?'Preparation is complete. The summary is below; use '+applyLabel(state.preview.plan)+' in the bottom bar to continue.':(online?'Downloads WanGP source':'Reads the selected source')+' and resolves compatible packages. Some package metadata or wheels may also download. Installation starts only when you confirm the prepared result.')+'</p>'+(!state.preview?button('preview',maintaining?'Prepare changes':online?'Download and prepare':'Prepare installation','primary',!ready||(!maintaining&&!form.root)||busyJob()||server()||state.runtime_error):'')+'</div></section></div></fieldset>';
}

function review(){
  const data=state.preview;if(!data)return "";const p=data.plan;
  return html('<section class="card review" id="installation-review"><div class="row"><h2>Prepared changes</h2><span class="pill purple">{operation}</span></div><p class="path">{root}</p><div class="review-summary"><div><span>Target version</span><strong>{version}</strong></div><div><span>Environment</span><strong>{environment}</strong></div><div><span>Source files</span><strong>{count}</strong></div></div><p class="hint">{checkpoint} Your application settings and media are preserved.</p><details><summary>Show affected files</summary><pre>{changes}</pre></details><div class="actions">{discard}</div></section>',{
    operation:esc(p.operation),root:esc(p.installation_root),version:esc(p.target_release_id),environment:p.reuse_environment?(p.operation==="rollback"?"Restore saved environment":"Reuse validated environment"):p.environment_kind==="conda"?"Update a separate Conda clone":"Build a new environment",count:p.changes.length+" change"+(p.changes.length===1?"":"s"),
    checkpoint:p.previous_release_id?"Current version "+esc(p.previous_release_id)+" will remain available as a rollback checkpoint.":"A new managed installation will be created.",
    changes:p.changes.slice(0,100).map(x=>(x.before?x.after?"Change":"Remove":"Add")+"  "+esc(x.path)).join("\n")+(p.changes.length>100?"\n… and "+(p.changes.length-100)+" more":""),
    discard:button("discard-preview","Discard prepared changes","quiet",busyJob())
  });
}
function maintenance(){
  const item=selected();
  if(!item)return heading("MAINTENANCE","Keep a way back.","Manage releases and restore a known-good version.")+'<div class="card empty">'+svg("maintain")+'<h2>Add an installation first</h2><p>Managed installations have version switching, repair and rollback.</p>'+button("navigate","Set up WanGP →","text-link",false,'data-page="install"')+'</div>';
  if(!canMaintain(item))return heading("MAINTENANCE","Existing installation.","This checkout remains under your control.")+notice("Manager can launch and inspect this installation. Use its existing tools to update source or packages, or create a separate managed installation to test version switching.")+button("navigate","Create a managed installation","primary",false,'data-page="install"');
  return heading("UPDATES","Update your WanGP installation.","Keep this installation and its models. Prepare an update, review it, then apply it.")+
    '<section class="card update-entry"><h2>Update WanGP to the latest version</h2><p>Currently installed: <strong>'+esc(item.release||'Existing checkout')+'</strong></p><p class="hint">Manager checks the latest version and its dependencies, then keeps a rollback checkpoint. Your models, plugins, settings and outputs stay in place.</p>'+button("prepare-latest","Prepare latest update","primary",server()||busyJob()||item.pending_operation||state.runtime_error)+'</section>'+review()+
    (usesConda(item)?notice("Conda updates reuse your current environment when it satisfies the selected dependencies. When packages need to change, Manager prepares and validates a separate clone. Stop WanGP and save or stash tracked source edits before updating."):"")+
    (server()?notice("Stop this installation before changing its version or environment."):"")+
    (item.pending_operation?notice("An interrupted operation needs recovery before this installation can launch. "+button("recover","Recover now"),"error"):"")+
    '<details class="card version-options"><summary>Choose another version or change dependencies</summary>'+installForm(true)+'</details><section class="card"><div class="row"><h2>Saved versions and rollback</h2>'+button("versions","Refresh history","small",server()||busyJob())+'</div>'+
    (state.versions.length?'<ul class="list">'+state.versions.map(savedVersionRow).join("")+'</ul>':'<p class="hint">Your first update will save the current version here for rollback.</p>')+'</section>';
}

function savedVersionRow(version){
  const attr='data-checkpoint="'+esc(version.checkpoint)+'"';
  const disabled=server()||busyJob()||selected().pending_operation;
  let description="Active source and environment",actions='<span class="pill green">Current</span>';
  if(!version.current){
    if(version.removing||currentFlow().removeCheckpoint===version.checkpoint){
      description=version.removing?"Removal is incomplete. Close any program using the old environment and retry.":"Remove this rollback version? Manager also deletes its environment if it owns it and no other version or registered installation uses it.";
      actions=button("remove-version",version.removing?"Finish removal":"Confirm removal","small danger",disabled,attr);
      if(!version.removing)actions+=button("cancel-removal","Cancel","small",disabled);
    }else{
      description="Retained rollback checkpoint";
      actions=button("rollback","Preview restore","small",disabled,attr)+button("confirm-removal","Remove","small",disabled,attr);
    }
  }
  return html('<li><div class="version-main"><span class="version-node {current}"></span><div><strong>{release}</strong><p>{description}</p></div></div><div class="actions">{actions}</div></li>',{current:version.current?"current":"",release:esc(version.release_id),description,actions});
}
function diagnostics(){
  const hardware=state.hardware,info=state.inspection,packages=info?.inspection?.packages||{};
  return heading("DIAGNOSTICS","Know what is ready.","Inspect this computer and the selected installation.",button("refresh-diagnostics","Refresh checks"))+
    html('<div class="grid2"><section class="card"><div class="row"><h2>Manager runtime</h2><span class="pill {runtimeClass}">{runtime}</span></div><p class="hint">Bundled Python and uv are checked before the worker starts.</p><div class="path-block path">{data}</div></section><section class="card"><h2>This computer</h2><div class="metric-value">{ram}</div><p class="hint">{platform}</p></section></div><div class="spacer"></div><section class="card"><h2>Detected GPUs</h2><ul class="list">{gpus}</ul><p class="footer-note">Hardware detection does not establish GPU execution or generation quality.</p></section><section class="card"><div class="row"><h2>Selected environment</h2><span class="pill">{status}</span></div><p class="path">{python}</p><p class="metric-detail">{environment}</p>{error}{packages}<p class="footer-note">These are interpreter and package metadata checks. Server readiness and generation validation are separate.</p></section><div class="actions">{export}<span class="hint">Includes local paths and operation logs. Saved locally; never uploaded.</span></div>',{
      runtimeClass:state.runtime_error?"amber":"green",runtime:state.runtime_error?"Needs attention":"Verified",data:esc(state.data_directory),
      ram:hardware?.ram_bytes?Math.round(hardware.ram_bytes/1024**3)+" GB RAM":"Waiting for hardware check",platform:esc(hardware?[hardware.platform,hardware.machine].join(" · "):""),
      gpus:(hardware?.devices||[]).map(x=>'<li><div><strong>'+esc(x.name)+'</strong><p>'+esc(x.id)+'</p></div><span class="pill">'+(x.memory_bytes?Math.round(x.memory_bytes/1024**3)+" GB":esc(x.vendor))+'</span></li>').join("")||'<li><p class="hint">No device inventory yet. Run Refresh checks.</p></li>',
      environment:esc(environmentDetails(info?.inspection)),status:esc(info?.inspection?.status||"Not checked"),python:esc(info?.python||selected()?.python||"Add an installation to inspect its environment."),error:info?.inspection?.error?notice(esc(info.inspection.error),"error"):"",
      packages:Object.keys(packages).length?'<table><thead><tr><th>Package</th><th>Installed version</th></tr></thead><tbody>'+Object.entries(packages).filter(([,v])=>v).map(([k,v])=>'<tr><td>'+esc(k)+'</td><td>'+esc(v)+'</td></tr>').join("")+'</tbody></table>':"",export:button("export","Export diagnostic report…")
    });
}
function environmentDetails(info){
  if(!info)return "Checking environment…";
  if(info.status!=="inspected")return info.error||"Environment unavailable";
  return ["Python "+info.python,"PyTorch "+(info.packages.torch||"not installed"),"CUDA "+(info.cuda||(info.packages.torch?"not included in this PyTorch build":"not installed")),...(info.conda_environment?["Conda: "+info.conda_environment]:[])].join("\n");
}
function jobLog(job){return job.events.map(e=>e.text||([e.title,e.message].filter(Boolean).join(" — ")||(e.type==="finished"?"Operation "+(e.result?.status||"completed")+(e.result?.error?": "+e.result.error.message:""):e.type==="server_ready"?"Server ready: "+e.url:""))+"\n").join("").replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g,"").slice(-180000);}
function consoleMarkup(job,id,placeholder){
  const attributes='data-id="'+(job?.id||"")+'"';
  const actions=[
    iconButton("copy-console","copy","Copy console",!job,attributes),
    iconButton("save-console","save","Save console",!job,attributes),
    state.consoleExpanded ? iconButton("minimize-manager","minimize","Minimize Manager") : "",
    iconButton("toggle-console",state.consoleExpanded?"collapse":"expand",state.consoleExpanded?"Exit full-screen console (Esc)":"Full-screen console",false,'id="console-size"')
  ].join("");
  return '<section class="console'+(state.consoleExpanded?' console-fullscreen':'')+'" aria-label="Console"><div class="row"><h3>Console</h3><div class="console-actions">'+actions+'</div></div><pre id="'+id+'" class="log" tabindex="0" aria-label="WanGP console">'+esc(job?jobLog(job):placeholder)+'</pre><p class="hint">Recent console output · Copy and Save include the text shown here.'+(state.consoleExpanded?' · Press Esc to return.':'')+'</p>'+(state.consoleExpanded&&state.message?'<p class="console-feedback" role="status">'+esc(state.message.text)+'</p>':"")+'</section>';
}
async function setConsoleExpanded(enabled){
  const log=document.getElementById("server-log");
  const position=log?{top:log.scrollTop,bottom:log.scrollTop+log.clientHeight>=log.scrollHeight-30}:null;
  await invoke("set_console_fullscreen",{enabled});
  state.consoleExpanded=enabled;render();
  const restored=document.getElementById("server-log");
  if(position&&restored)restored.scrollTop=position.bottom?restored.scrollHeight:position.top;
  document.getElementById("console-size")?.focus({preventScroll:true});
}
function activity(){
  const jobs=[...state.jobs.values()].reverse(),active=jobs.find(x=>x.id===state.selectedLog)||jobs[0];
  return heading("ACTIVITY","Every step, in view.","Follow operations and inspect their results.")+
    '<section class="card"><div class="field"><label for="log-picker">Operation</label><select id="log-picker">'+(jobs.map(x=>option(x.id,x.action+" · "+(x.done?x.result?.status||"Complete":"Running")+" · "+x.id.slice(0,8),active?.id)).join("")||'<option>No operations yet</option>')+
    '</select></div><pre id="live-log" class="log">'+esc(active?jobLog(active):"Operations will appear here as you use Manager.")+'</pre>'+
    (active&&!active.done&&active.action!=="remove_version"?'<div class="actions">'+button("cancel",active.action==="launch"?"Stop server":"Cancel operation","danger",false,'data-id="'+active.id+'"')+'</div>':"")+
    (active?.result?'<details><summary>Result details</summary><pre>'+pretty(active.result)+'</pre></details>':"")+'</section><p class="footer-note">The live view keeps recent events. Durable installation journals remain with each managed installation.</p>';
}
function savedLaunchOptions(){return {listen:false,verbose:false,authentication:false,password_mode:"generated",debug_deepy:false,telemetry:false,...state.preferences.launch_options,debug_directory:state.preferences.launch_options?.debug_directory||state.data_directory+"/logs/deepy",telemetry_file:state.preferences.launch_options?.telemetry_file||state.data_directory+"/logs/telemetry.log"};}
function preferencesDirty(){return !!state.authenticationDraft.password||String(state.preferencesDraft?.port??state.preferences.port)!==String(state.preferences.port)||(state.preferencesDraft?.app_without_gpu??!!state.preferences.app_without_gpu)!==!!state.preferences.app_without_gpu||!!state.launchDraft&&Object.entries(savedLaunchOptions()).some(([key,value])=>state.launchDraft[key]!==value);}
function updatePreferenceStatus(){
  const dirty=preferencesDirty(),save=document.querySelector('[data-action="save-launch-options"]'),discard=document.querySelector('[data-action="discard-preferences"]'),status=document.getElementById('preferences-status');
  if(save)save.disabled=!dirty||state.savingPreferences;
  if(discard)discard.disabled=!dirty||state.savingPreferences;
  if(status)status.textContent=state.savingPreferences?'Saving…':dirty?'Unsaved changes — saved settings still apply until you save.':'All preferences saved.';
}
function preferences(){
  state.launchDraft??=savedLaunchOptions();
  state.preferencesDraft??={port:state.preferences.port,app_without_gpu:!!state.preferences.app_without_gpu};
  const draft=state.launchDraft;
  const launchOption=(key,label,help)=>'<div class="field"><label class="check" for="launch-'+key+'"><input id="launch-'+key+'" type="checkbox" data-group="launchDraft" data-key="'+key+'" '+(draft[key]?'checked':'')+'>'+label+'</label><p class="hint">'+help+'</p></div>';
  return heading("PREFERENCES","Make yourself at home.","Changes apply the next time you start WanGP.")+
    '<fieldset class="preferences-fields" '+disabled(state.savingPreferences)+'><section class="card"><h2>Launching</h2><p class="hint">Open in App and Open in Browser connect to the same running server.</p><div class="spacer"></div><div class="field"><label for="port">Server port</label><input id="port" type="number" min="1024" max="65535" data-group="preferencesDraft" data-key="port" value="'+esc(state.preferencesDraft.port)+'"></div>'+
    launchOption('listen','Accessible on the local network','Connect from another device on your network. Adds --listen.')+'</section>'+
    '<section class="card"><h2>App window</h2><div class="field"><label class="check" for="app-without-gpu"><input id="app-without-gpu" type="checkbox" data-group="preferencesDraft" data-key="app_without_gpu" '+(state.preferencesDraft.app_without_gpu?'checked':'')+'>Run the App window without the graphics card</label>'+
    '<p class="hint">Leaves the graphics memory the App window would use to WanGP. Pages and video previews are then drawn by the processor, so scrolling and playback can be less smooth. Open in Browser is not affected: your browser has its own setting.</p>'+
    (!!state.preferences.app_without_gpu!==!!state.app_without_gpu_active?'<p class="hint" role="status"><strong>Restart Manager to apply this change.</strong></p>':'<p class="hint">Applies when Manager starts.</p>')+'</div></section>'+
    '<section class="card"><h2>Authentication</h2>'+launchOption('authentication','Require a password','Optional password protection for the WanGP web interface.')+
    (draft.authentication?'<div class="field"><label for="password-mode">Password</label><select id="password-mode" data-group="launchDraft" data-key="password_mode">'+option('generated','Generate a password at every launch',draft.password_mode)+option('custom','Use my own password',draft.password_mode)+'</select></div>'+
    (draft.password_mode==='custom'?'<div class="field"><label for="custom-password">Custom password</label><input id="custom-password" type="password" autocomplete="new-password" data-group="authenticationDraft" data-key="password" value="'+esc(state.authenticationDraft.password)+'" placeholder="'+(state.preferences.password_saved?'Password saved — leave blank to keep it':'Enter your password')+'"><p class="hint">'+(state.platform==='linux'?'Saved in your desktop keyring.':'Saved with Windows protection for your user account.')+'</p></div>':'<p class="hint">WanGP generates a fresh password and displays it in the launch console.</p>'):'')+'</section>'+
    '<section class="card"><h2>Debug</h2>'+launchOption('verbose','Detailed console','Adds --verbose 2.')+
    launchOption('debug_deepy','Deepy debug logs','Adds --debug-deepy and enables verbose level 2.')+
    (draft.debug_deepy?field('launchDraft','debug_directory','Deepy log folder','Choose where Deepy saves its debug logs','folder'):'')+
    launchOption('telemetry','Deepy telemetry','Enable detailed decode, MTP, CUDA-memory and GPU diagnostics. Also enables verbose level 2 and saves console output locally; nothing is uploaded.')+
    (draft.telemetry?field('launchDraft','telemetry_file','Telemetry output file','Choose a .log or .txt file','log')+'<p class="hint">Output is appended to this file while WanGP runs. Detailed decode telemetry appears when Deepy is used.</p>':'')+'</section>'+
    '<div class="preference-save"><div class="actions">'+button('save-launch-options',state.savingPreferences?'Saving…':'Save preferences','primary',!preferencesDirty()||state.savingPreferences)+button('discard-preferences','Discard changes','quiet',!preferencesDirty()||state.savingPreferences)+'</div><p id="preferences-status" class="hint" role="status">'+(state.savingPreferences?'Saving…':preferencesDirty()?'Unsaved changes — saved settings still apply until you save.':'All preferences saved.')+'</p><p class="hint">Existing scripts/args.txt settings still apply unless an enabled Manager option overrides them.</p></div></fieldset>'+
    '<section class="card"><h2>Update WanGP Manager</h2><p>Installed Manager: <strong>'+esc(state.version)+'</strong></p>'+(state.managerUpdate?'<p role="status">'+(newerVersion(state.managerUpdate.version,state.version)?'Manager '+esc(state.managerUpdate.version)+' is available.':'You have the latest Manager version.')+'</p>':'')+'<p class="hint">Download the latest installer, close Manager, then run it. It updates Manager in place and keeps your installation list and preferences. You do not need to uninstall first. WanGP itself is updated from Update WanGP in the sidebar.</p><div class="actions">'+button('manager-update','Check for Manager updates','',busyJob())+button('manager-download','Download latest Manager')+'</div></section>'+ '<aside class="manager-information" aria-label="Manager information"><h2>Manager information</h2><p class="hint">Information only. These locations are managed automatically and are not editable preferences.</p><dl><dt>App version</dt><dd>WanGP Manager '+esc(state.version)+'</dd><dt>Installation list and preferences</dt><dd><code>'+esc(state.data_directory.replace(/[\\/]+$/,'')+'/preferences.json')+'</code></dd><dt>Per-installation records</dt><dd>The .manager folder inside each managed installation stores its active version, environment and rollback history.</dd></dl><p class="hint">Add existing installations and create test installations from the Install / Add page. Generation workspaces remain inside WanGP.</p></aside>';
}
function newerVersion(candidate,current){
  const a=candidate.split('.').map(Number),b=current.split('.').map(Number);
  for(let i=0;i<Math.max(a.length,b.length);i++){if((a[i]||0)!==(b[i]||0))return (a[i]||0)>(b[i]||0);}
  return false;
}

function notify(text,type="success"){state.message={text,type};render();}
async function run(action,args={}){
  const id=await invoke("start_job",{action,args}),existing=state.jobs.get(id);
  state.jobs.set(id,{id,action,done:false,events:[],...existing,args});render();
  if(existing?.done){if(existing.result?.error||["failed","cancelled","recovery_required"].includes(existing.result?.status))throw new Error(existing.result.error?.message||existing.result.status);return existing.result;}
  return await new Promise((resolve,reject)=>state.pending.set(id,{resolve,reject}));
}
async function readCatalog(){
  const args=sourceArgs(),flow=currentFlow();
  flow.catalog=null;flow.preview=null;
  const catalog=await run("catalog",args);
  flow.catalog=catalog;
  if(flow===currentFlow()){inferProfile();render();}
}
async function loadCondaEnvironments(){
  if(state.conda.loading||state.runtime_error)return;
  state.conda.loading=true;state.conda.error=null;render();
  try{
    const result=await run("conda_environments");
    Object.assign(state.conda,{environments:result.environments,warnings:result.warnings,loaded:true});
  }catch(error){state.conda.error=error.message||String(error);}
  finally{state.conda.loading=false;render();}
}
async function detectInstallation(){
  const root=state.registration.root;if(!root)return;
  const request={environments:[],loading:true,error:null};
  state.registration.python="";state.detected=request;render();
  try{
    const result=await run("detect_installation",{root});
    if(root!==state.registration.root||state.detected!==request)return;
    state.registration.root=result.root;state.registration.python=result.python||"";
    state.detected.environments=result.environments;
  }catch(error){if(root===state.registration.root&&state.detected===request)request.error=error.message||String(error);}
  finally{if(state.detected===request){request.loading=false;render();}}
}
async function prepareUpdate(){
  state.message=null;state.preview=null;
  const flow=currentFlow();
  const args={...state.form,...sourceArgs(),root:state.page==="maintenance"?selected().root:state.form.root,python:state.page==="maintenance"?selected().python:null,operation:state.page==="maintenance"?state.form.operation:"install",environment:state.page==="maintenance"?state.form.environment:"auto"};
  flow.preview=await run("preview",args);render();
}
function inferProfile(){
  const profiles=state.catalog?.profiles||{},devices=state.hardware?.devices||[];
  const supported=device=>profiles[device.profile]?.platforms.includes(state.hardware.platform);
  if(profiles.Demo){state.form.profile="Demo";state.form.device="";return;}
  const gpu=devices.find(x=>x.id===state.form.device)||devices.find(supported)||devices[0];
  state.form.device=gpu?.id||"";state.form.profile=gpu&&supported(gpu)?gpu.profile:"";
}
async function loadSources(refresh=false,older=false){
  const flow=currentFlow(),remote=flow.remote,repository=remote.repository,page=older?remote.page+1:1;
  await state.latestCheck;  // both write the same catalog cache: Windows refuses a file replaced while another process replaces it
  const result=await run("sources",{repository,refresh,page,catalog_id:older?remote.catalog_id:null});
  if(repository!==remote.repository)return;
  if(older){
    const seen=new Set(remote.choices.map(x=>x.commit));
    remote.choices.push(...result.versions.filter(x=>!seen.has(x.commit)));
  }else{
    remote.choices=result.versions;
    if(!result.versions.some(x=>x.commit===remote.commit))remote.commit=result.versions[0]?.commit||"";
  }
  Object.assign(remote,{page:result.page,has_more:result.has_more,catalog_id:result.catalog_id,origin:result.catalog_origin,stale:result.stale,loaded_repository:repository});
  render();
  if(flow===currentFlow()&&remote.commit&&!older)await readCatalog();
}
async function refreshInspection(){
  const item=selected();if(!item)return;
  const inspection=await run("inspect",{root:item.root,python:item.python});
  if(selected()?.root!==item.root)return;
  state.inspection=inspection;
  const current=state.preferences.installations.find(x=>x.root===item.root);if(current)Object.assign(current,state.inspection);
  render();
}
async function history(){
  if(!canMaintain(selected())||server())return;
  const root=selected().root,result=await run("versions",{root,maintenance:selected().maintenance||null});
  if(selected()?.root===root){state.versions=result.versions;render();}
}
async function navigate(page){
  state.page=page;state.registerView=false;state.message=null;selectFlow(page);
  if(state.catalog&&["install","maintenance"].includes(page))inferProfile();
  render();
  if(page==="maintenance"){await refreshInspection();await history();}
  if(state.page!==page)return;
  if((page==="install"||page==="maintenance")&&!busyJob()){
    if(usingRemote()&&!state.catalog)await loadSources();
    else if(!usingRemote()&&state.form.source&&!state.catalog)await readCatalog();
  }
  if(page==="overview")await refreshInspection();
  if(page==="diagnostics"){if(!state.hardware)state.hardware=await run("hardware");await refreshInspection();}
}
document.addEventListener("input",event=>{
  const {group,key}=event.target.dataset;
  if(group&&key){
    state[group][key]=event.target.type==="checkbox"?event.target.checked:event.target.value;
    if(group==="registration"&&key==="root"){state.registration.python="";state.detected={environments:[],loading:false,error:null};}
    if(group==="form"||group==="remote")state.preview=null;
    if((group==="form"&&key==="source")||group==="remote"&&["kind","commit","repository"].includes(key))state.catalog=null;
    if(group==="remote"&&key==="repository"){state.remote.choices=[];state.remote.commit="";state.remote.loaded_repository=null;}
    if(["preferencesDraft","launchDraft","authenticationDraft"].includes(group))updatePreferenceStatus();
  }
});
document.addEventListener("change",async event=>{
  try{
    if(event.target.id==="installation-picker"){await invoke("select_installation",{root:event.target.value});state.preferences.selected=event.target.value;state.inspection=null;state.versions=[];await navigate(state.page);}
    else if(event.target.id==="log-picker"){state.selectedLog=event.target.value;render();}
    else if(["conda-environment","local-environment"].includes(event.target.id)){state.registration.python=event.target.value;render();}
    else if(event.target.dataset.group==="registration"&&event.target.dataset.key==="root")await detectInstallation();
    else if(event.target.dataset.group==="remote"){
      if(event.target.dataset.key==="manual"&&!state.remote.manual){Object.assign(state.remote,{kind:"github",repository:OFFICIAL_REPOSITORY,commit:"",choices:[],loaded_repository:null});state.catalog=null;}
      render();
      if(usingRemote()&&event.target.dataset.key==="commit")await readCatalog();
      else if(usingRemote()&&["kind","manual"].includes(event.target.dataset.key)&&!state.catalog)await loadSources();
      else if(!usingRemote()&&state.form.source)await readCatalog();
    }
    else if(event.target.dataset.key==="device"){inferProfile();state.preview=null;render();}
    else if(event.target.dataset.group)render();
  }catch(error){notify(error.message||String(error),"error");}
});
document.addEventListener("click",async event=>{
  const el=event.target.closest("[data-action]");if(!el||el.disabled)return;
  const action=el.dataset.action;
  try{
    if(action==="navigate")await navigate(el.dataset.page);
    else if(action==="dismiss"){state.message=null;render();}
    else if(action==="register-view"){state.page="install";selectFlow("install");state.registerView=true;render();if(!state.conda.loaded)await loadCondaEnvironments();}
    else if(action==="refresh-conda")await loadCondaEnvironments();
    else if(action==="detect-installation")await detectInstallation();
    else if(action==="manager-update"){state.managerUpdate=await run("manager_update");render();}
    else if(action==="manager-download")await invoke("open_manager_download");
    // WanGP asks for the star itself; the next launch tells it the project page was opened here.
    else if(action==="github-star"){state.preferences=await invoke("github_star");render();}
    else if(action==="new-view"){state.registerView=false;render();}
    else if(action==="browse"){
      const value=await invoke("pick_path",{kind:el.dataset.kind});
      if(value){state[el.dataset.group][el.dataset.key]=value;if(el.dataset.group==="form"||el.dataset.group==="registration")state.preview=null;render();if(el.dataset.group==="form"&&el.dataset.key==="source")await readCatalog();if(el.dataset.group==="registration"&&el.dataset.key==="root")await detectInstallation();}
    }
    else if(action==="catalog")await readCatalog();
    else if(action==="sources")await loadSources(true);
    else if(action==="older-sources")await loadSources(false,true);
    else if(action==="register"){await run("register",{...state.registration});if(state.page==="install"&&state.registerView){state.registerView=false;state.page="overview";}notify("Installation added. Its source and packages were left unchanged.");}
    else if(action==="demo"){
      state.demo=await run("demo");selectFlow("install");Object.assign(state.remote,{manual:true,kind:"local"});Object.assign(state.form,{source:state.demo.releases[0].path,root:state.demo.root,profile:"Demo",device:"",extras:true,operation:"install"});
      state.page="install";state.registerView=false;state.preview=null;await readCatalog();notify("Test source ready. Choose Prepare installation, then Install test installation to try the sample workflow.");
    }
    else if(action==="demo-version"){state.form.source=el.dataset.source;state.preview=null;await readCatalog();}
    else if(action==="preview")await prepareUpdate();
    else if(action==="prepare-latest"){
      Object.assign(state.remote,{manual:false,kind:"github",repository:OFFICIAL_REPOSITORY,commit:""});
      state.form.operation="update";state.form.environment="auto";state.preview=null;
      await loadSources(true);await prepareUpdate();
    }
    else if(action==="apply"){
      const flow=currentFlow(),page=state.page;
      const result=await run("apply",{preview_id:flow.preview.preview_id});flow.preview=null;
      if(state.page===page&&currentFlow()===flow)state.page="overview";
      notify("Installation changes completed. The installed version is ready to launch.");if(result.installation&&state.page==="overview")await refreshInspection();
    }
    else if(action==="toggle-progress"){state.progressExpanded=!state.progressExpanded;render();}
    else if(action==="discard-preview"){state.preview=null;render();}
    else if(action==="versions")await history();
    else if(action==="confirm-removal"){currentFlow().removeCheckpoint=el.dataset.checkpoint;render();}
    else if(action==="cancel-removal"){currentFlow().removeCheckpoint=null;render();}
    else if(action==="remove-version"){
      const root=selected().root,flow=currentFlow();
      flow.preview=null;flow.removeCheckpoint=null;
      try{
        const result=await run("remove_version",{root,checkpoint:el.dataset.checkpoint});
        if(selected()?.root===root)notify(result.environment_removed?"Saved version and unused environment removed.":"Saved version removed. Its environment was kept.");
      }finally{if(selected()?.root===root)await history();}
    }
    else if(action==="rollback"){const flow=currentFlow();flow.preview=await run("preview",{root:selected().root,operation:"rollback",checkpoint:el.dataset.checkpoint});render();}
    else if(action==="recover"){
      const result=await run("recover",{root:selected().root});await refreshInspection();
      notify(result.status==="failed"?"Recovery completed. The interrupted operation was not applied.":"Installation recovery completed.");
    }
    else if(action==="start"){const item=selected();run("launch",{root:item.root,python:item.python,port:state.preferences.port}).then(result=>{if(result.status==="crashed")notify("WanGP stopped unexpectedly (exit "+result.returncode+"). Open Activity for its logs.","error");else render();}).catch(error=>notify(error.message,"error"));}
    else if(action==="stop"){state.message=null;await invoke("cancel_job",{id:server().id});}
    else if(action==="open-app"||action==="open-browser")await invoke("open_server",{id:server().id,presentation:action==="open-app"?"app":"browser"});
    else if(action==="toggle-console")await setConsoleExpanded(!state.consoleExpanded);
    else if(action==="minimize-manager")await window.__TAURI__.window.getCurrentWindow().minimize();
    else if(action==="copy-console"||action==="save-console"){
      const text=jobLog(state.jobs.get(el.dataset.id)),save=action==="save-console";
      const path=await invoke("console_text",{text,save});
      if(!save)notify("Console copied. Paste it wherever you need it.");
      else if(path)notify("Console saved: "+path);
    }
    else if(action==="cancel"){await invoke("cancel_job",{id:el.dataset.id});notify("Cancellation requested. Manager is stopping its owned processes.");}
    else if(action==="refresh-diagnostics"||action==="refresh-hardware"){const previousProfile=state.form.profile;state.hardware=await run("hardware");inferProfile();if(previousProfile!==state.form.profile)state.preview=null;if(action==="refresh-diagnostics")await refreshInspection();render();}
    else if(action==="discard-preferences"){state.launchDraft=null;state.preferencesDraft=null;state.authenticationDraft.password="";state.message=null;render();}
    else if(action==="save-launch-options"){
      const port=Number(state.preferencesDraft.port);
      if(!Number.isInteger(port)||port<1024||port>65535)throw new Error("Choose a port between 1024 and 65535.");
      state.savingPreferences=true;render();
      try{
        state.preferences=await invoke("set_launch_options",{options:{...state.launchDraft},password:state.authenticationDraft.password||null,port,appWithoutGpu:!!state.preferencesDraft.app_without_gpu});
        state.authenticationDraft.password="";state.launchDraft=savedLaunchOptions();state.preferencesDraft={port:state.preferences.port,app_without_gpu:!!state.preferences.app_without_gpu};
        notify("Preferences saved. They will apply the next time you start WanGP.");
      }finally{state.savingPreferences=false;render();}
    }
    else if(action==="export"){const path=await invoke("export_diagnostics");if(path)notify("Diagnostic report saved: "+path);}
    else if(action==="keep-open")document.querySelector("#close-dialog").close();
    else if(action==="exit"){el.disabled=true;el.textContent="Stopping…";await invoke("stop_and_exit");}
  }catch(error){notify(error.message==="cancelled"?"Operation cancelled.":error.message||String(error),error.message==="cancelled"?"":"error");}
});
document.addEventListener("keydown",event=>{
  if(event.key==="Escape"&&state.consoleExpanded){event.preventDefault();setConsoleExpanded(false).catch(error=>notify(error.message||String(error),"error"));}
});
await listen("manager:job",({payload:{id,action,event}})=>{
  let job=state.jobs.get(id);if(!job){job={id,action,done:false,events:[],args:{}};state.jobs.set(id,job);}
  job.events.push(event);if(job.events.length>250)job.events.shift();
  if(event.type==="server_ready"){job.server_url=event.url;render();}
  if(event.type==="server_starting"){job.server_url=null;render();}
  if(event.type==="server_stopping"){job.server_url=null;job.stopping=true;render();}
  if(event.type==="finished"){
    job.done=true;job.result=event.result;job.server_url=null;
    if(action==="launch"&&event.result?.status==="stopped")state.message={text:"Server stopped.",type:"success"};
    const pending=state.pending.get(id);state.pending.delete(id);
    if(pending){if(event.result?.error||["failed","cancelled","recovery_required"].includes(event.result?.status))pending.reject(new Error(event.result.error?.message||event.result.status));else pending.resolve(event.result);}
    render();
  }else{
    updateOperationProgress();
    if(state.page==="activity"&&(!state.selectedLog||state.selectedLog===id)){const log=document.querySelector("#live-log");if(log){const bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-30;log.textContent=jobLog(job);if(bottom)log.scrollTop=log.scrollHeight;}}
    if(state.page==="overview"&&action==="launch"&&job.args?.root===selected()?.root){const log=document.querySelector("#server-log");if(log){const bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-30;log.textContent=jobLog(job);if(bottom)log.scrollTop=log.scrollHeight;}}
  }
});
await listen("manager:registry",({payload})=>{state.preferences=payload;render();});
await listen("manager:close-requested",()=>document.querySelector("#close-dialog").showModal());
try{
  const initial=await invoke("get_state");Object.assign(state,initial);state.jobs=new Map(initial.jobs.map(x=>[x.id,x]));state.form.root=initial.default_installation;render();
  if(!state.runtime_error)run("hardware").then(value=>{state.hardware=value;inferProfile();render();}).catch(error=>notify(error.message,"error"));
  if(!state.runtime_error)checkUpdates();
  if(!state.runtime_error&&selected())refreshInspection().catch(error=>notify(error.message,"error"));
  if(!state.runtime_error&&!selected())loadSources().catch(error=>notify(error.message,"error"));
}catch(error){document.querySelector("#app").textContent="Manager could not start: "+(error.message||String(error));}
