# Loras

#### Using Lora Multipliers in WanGP

You may know LoRAs—but WanGP uses them differently. This guide explains how they actually behave in this system, not just how they’re labeled in the UI.

In my own experience, Lora multipliers were more complex than I had first thought. I wasn't able to find a good guide on how to control them best, so I decided to make one and share it. Hopefully someone else will find it as useful as I would have. I wasn't even aware of all the levels of control I could exert so I give a little summary for each topic in case you also need it. I hope all this helps you take control of your Loras and run some experiments with the granular control you can take over their strength and overall effect on your videos.



For the purpose of applicability this guide is written for wan2.2, ie: a dual phase Lora setup. It's more complex and needs a different use entirely so you can apply this to single step Loras but not the other way around. 

Each Lora should include two files, unlike those for wan2.1/etc.  When you download a Lora you need:

- A High Noise Model (usually a .safetensors file that says "high" or just "H" in the filename)

- A Low Noise Model (usually a .safetensors file that says "low" or just "L" in the filename)



In WanGP the strength of a Lora and granular control over how those Loras affect different areas of video generation is more complex than it seems at first, and is primarily determined by the values entered into two boxes:

"Activated Loras"


and 

"Loras Multipliers (1.0 by default) separated by Space chars or CR, lines that start with # are ignored"



##### Activated Loras

<u>This part is mandatory</u>. Thankfully it's fairly straightforward but there are still a few things to keep in mind. 

- Make sure you have the matching High and Low model for the Lora you want to use. Mixing models has 'mixed' results, so use the two that come as a pair.
- Decide which Lora you want to use in advance, if you want to add one later you will have to remove and reattach some of the other files you have first. 
- ⚠Always attach Lora files in the order from **HIGH to LOW** noise. If you use multiple Loras, you need to attach **<u>all</u> HIGH Noise models**, before you attach their counterparts, the **all LOW Noise **.
- ⚠Always attach the low noise models in the same order that you attached the high noise models.



##### Loras Multipliers

This part is <u>optional</u> but where the real complexity comes in. If you want more granular control over the Loras you activated above, you need to know the syntax and what you can do with it.

There are a few concepts which relate to what you can control here. 

- Each value you enter is a  % of the Lora's total standard weight or **strength** that it can have on generation. The default is 1.0 and you don't need to enter anything here if you just wanted 100% of the activated Lora(s) strength. 
- This can easily be just one number, say 0.5 , and activated Lora(s) will have a 50% of their full strength. If that's your goal, nothing else is needed.



But in addition to that you can control:

- <u>The number of "Phases"</u> of generation that take place, and how much impact the Lora have on it. Models that are dual phase like Wan2.2 use a baseline of those two phases. You have three phases if you wanted to or more. Why? The best and simplest way to explain it is like this: 

  **2 phases:**   (1) Generate motion (High) →  (2) refine quality (Low)

  **3 phases:**   (1) Generate motion / base clip → (2) extend the time of the video  →  (3) upscale  and refine the quality

  **4 phases+**:  (1) Base Clip → (2) Time extension →  (3) add some new feature → and so on up to the final refinement phase. 

  There's no limit, you're splitting up the denoising process into more numbers of smaller steps to add control. This doesn't lengthen the total process it just divides it up into more smaller parts. As we will see the more you chose to specify, the more complex the syntax becomes. 



Even deeper than than, you can control: 

- <u>The "Time-based Multipliers"</u>, as is it called, for each phase. This pretty much means specifying how many steps of your video generation each Phase will be given and the Lora's impact during precisely those parts of each denoising Phase. 



#### Entering Values into WanGP

This is the structure of the hierarchy should you opt to go a level deeper from each one above it.

Level 1 - Specify a value for each Lora File

Level 2 - Specify a value for each Phases of each Lora 

Level 3 - Specify a value for each Time Based Multiplier for each Phase of each Lora

Each level replaces the previous one with a more detailed breakdown—you are not adding values, you are subdividing them. Together they establish the syntax. Here is a visual way I used to understand how to program the values for each of these four element **In WanGP**: 



###### Level 1 - Lora File (simplest setup)

Format  **LORA1-high LORA2-high LORA1-low LORA2-low**	... and so on, highs first then lows.

​	→ EX:  H L
​	→ EX:  H H L L
​	→ EX:  H H H... L L L...

​	Example for two Loras, each with two files (4 values):  **0.5 0.2 1.0 1.4**

​	Rules:

​	⬩ Each Lora file is one value. 

​	⬩ Each Lora file value is separated by a SPACE character.

​	⬩ Lora files are always High then Low models

​	⬩ The order of the values match the exact order of the Lora files you activated.



###### Level 2 - Splitting up generation into more Phase (more complex). 

Each Lora file can be split into as many Phases as you want, but that number of phases will now affect ALL Loras you use. So if you want 3 Phases instead of two you break it down for each Lora. From the example above that looks like this. L is Lora and PH is phase:

​	      --------L1-------- --------L2-------- --------L3-------- --------L4--------

​	▶ PH1;PH2;PH3 PH1;PH2;PH3 PH1;PH2;PH3 PH1;PH2;PH3  ◀

​	

Each Lora is now 4 values instead of just 1. For 4 Lora files as we had before we now have 12 values to use them at this level of control. 

Example for two Loras, each with two files, split into 3 Phases:  

**0.4;0.6;0.5 0.2;1.0;1.4 0.4;0.8;0.4 1.0;1.3;1.0**

​	Rules:

​	⬩ Each value is separated by a **Semi-Colon** within each Lora file

​	⬩ Each set of values for each Lora file is still separated by a **SPACE** character. 

​	⬩ Each Lora file must contain an **equal number of Phases**.



###### Level 3 - Splitting up Phases Time-based Multipliers (TBMs)

This level takes a phase and splits it up into equally divided number of Steps (rounding for even integers aside).

The total steps you set are how many are divided up into these multipliers. 

→ A 10 STEP VIDEO CAN BE BROKEN DOWN IN TO 2 GROUPS OF 5 GENERATION STEPS BY USING 2 TBMs
→ A 21 STEP VIDEO CAN BE BROKEN DOWN IN TO THREE GROUPS OF  7 STEPS BY USING 3 TBMs
→ TBMs APPLY THE WEIGHT DURING EACH OF THOSE SPECIFIC STEPS OF THE GENERATION PROCESS.



​	--------------L1-------------- --------------L2-------------- --------------L2-------------- --------------L4--------------****

​	--PH1--;--PH2--;--PH3-- --PH1--- --PH2--;--PH3-- --PH1--;--PH2--;--PH3-- --PH1--;--PH2--;--PH3--

 ▶ m1,m2;m1,m2;m1,m2 m1,m2;m1,m2;m1,m2 m1,m2;m1,m2;m1,m2 m1,m2;m1,m2;m1,m2 ◀



Example for two Loras, each with two files, split into 3 Phases, with each Phase having two Time Based Multipliers: 

**0.3,0.5;0.4,0.7;0.5,0.5 0.1,0.4;0.6,0.9;1.1,1.8 1.0,1.0;1.1,1.4;1.0,1.6 0.9,1.2;1.0,1.3;1.1,1.5**

Now we have 6 values per Lora file: 2 TB Multipliers for each Phase, and 3 Phases per Lora.

Each further Lora will add two more files, meaning 12 further values. But that is just for TWO multipliers per Phase, and you can set as many as you like and with TBMs you can do this so for one, some, or all Phases independently. You don't have to split up every Phase into steps.  You just replace the phase values you need to with the TBM's to use in that particular Phase using comma separated values. 

​	Rules:

​	⬩ Each Time Based Multiplier value is separated by a **Comma** within each phase.

​	⬩ Each set of values for each Phase is still separated by a **Semi-Colon**. 

​	⬩ Each group of values for each Lora file is separated by a **SPACE** character.

Additional things to be aware of:

- Using control "Guidance (CFG)" with Loras is tricky. 
  - I found that most Loras, apparently those built on Lightning models do not behave well with Guidance turned on. I would get big blobs of color with indistinguishable video somewhere in it.  
  - Without Guidance turned on (activated above 1.0) this meant that I also couldn't use Negative Prompts with Guidance. For the mostpart I adapted using the main prompt with things like  "do NOT..." and "ABSOLUTELY NO ____" but I also found I had to make them 'stronger' with !!!!!! the end or by wrapping brackets around the instructions - although I don't know for certain if either actually do anything, it may have just been other random factors at play. To be fair I experimented with it all but results were 'inconsistent' at best, so don't take my word for this part (!!!)
  - NAG - an alternate way to enforce a negative prompt has a different approach, and while you don't get guidance out of it you can actually enforce negative prompts by turning it up. 
- Splitting Loras by Video Time:
  - Unfortunately, none of these controls allow you to apply Loras to different time-stamps of a video. Phases are applied to ALL parts of a video equally, as every second of every video is generated in the same phases.
  - To use Lora1 for seconds 1 to 5 of your video, and then use Lora2 for seconds 6 to 10, you have to create two videos, each with the Lora you want to affect them. So for seconds 1 to 5 you create a 5 second video using Lora1. Then you select "extend video" and add another 5 seconds to it using Lora2. It's not perfect but its that or make totally unique videos and stitch them together manually. which I find ten times less efficient and consistent.


That's all I can think of for now, happy Lora-ing folks, I hoped this helped someone! 

