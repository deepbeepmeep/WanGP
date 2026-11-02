# FINETUNES

A Finetuned model is model that shares the same architecture of one specific model but has derived weights from this model. Some finetuned models have been created by combining multiple finetuned models.

As there are potentially an infinite number of finetunes, specific finetuned models are not known by default by WanGP. However you can create a finetuned model definition that will tell WanGP about the existence of this finetuned model and WanGP will do as usual all the work for you: autodownload the model and build the user interface.

You can create a WanGP finetune definition in two ways:
- **With the Finetune Creator / Editor** in the model toolbar. This is the recommended path when you want to create a finetune from the currently selected model, edit an existing finetune, import a shared finetune JSON file, or keep the current model settings as the finetune defaults.
- **Manually**, by writing or editing a JSON definition in the **finetunes/** subfolder. This remains useful for advanced definitions, bulk editing, sharing files outside the UI, or when you want to review the exact JSON structure.

WanGP finetune system can be also used to tweak default models : for instance you ca plugin top of an existing model some loras that will be always applied transparently.

Finetune models definitions are light json files that can be easily shared. You can find some of them on the WanGP *discord* server https://discord.gg/g7efUW9jGV

All the finetunes definitions files should be stored in the *finetunes/* subfolder.



## Create a new Finetune Model Definition Manually
All the finetune models definitions are json files stored in the **finetunes/** sub folder. All the corresponding finetune model weights when they are downloaded will be stored in the *ckpts/* subfolder and will sit next to the base models.

All the models used by WanGP are also described using the finetunes json format and can be found in the **defaults/** subfolder. Please don’t modify any file in the **defaults/** folder.

However you can use these files as starting points for new definition files and to get an idea of the structure of a definition file. If you want to change how a base model is handled (title, default settings, path to model weights, …) you may override any property of the default finetunes definition file by creating a new file in the finetunes folder with the same name. Everything will happen as if the two models will be merged property by property with a higher priority given to the finetunes model definition.

A definition is built from a *settings file* that can contains all the default parameters for a video generation. On top of this file a subtree named **model** contains all the information regarding the finetune (URLs to download model, corresponding base model id, ...).

You can obtain a settings file in several ways:
- In the subfolder **settings**, get the json file that corresponds to the base model of your finetune (see the next section for the list of ids of base models)
- From the user interface, select the base model for which you want to create a finetune and click **export settings**

Here are steps:
1) Create a *settings file*
2) Add a **model** subtree with the finetune description
3) Save this file in the subfolder **finetunes**. The name used for the file will be used as its id. It is a good practise to prefix the name of this file with the base model. For instance for a finetune named **Fast*** based on  Hunyuan Text 2 Video model *hunyuan_t2v_fast.json*. In this example the Id is *hunyuan_t2v_fast*.
4) Use **Refresh Model List** in the model toolbar, or restart WanGP if you edited files while the UI was not active.

## Architecture Models Ids
A finetune is derived from a base model and will inherit all the user interface and corresponding model capabilities, here are some Architecture Ids:
- *t2v*: Wan 2.1 Video text 2 video
- *i2v*: Wan 2.1 Video image 2 video 480p and 720p
- *vace_14B*: Wan 2.1 Vace 14B
- *hunyuan*: Hunyuan Video text 2 video
- *hunyuan_i2v*: Hunyuan Video image 2 video

Any file name in the defaults subfolder (without the json extension) corresponds to an architecture id.

Please note that weights of some architectures correspond to a combination of weight of a one architecture which are completed by the weights of one more or modules.

A module is a set a weights that are insufficient to be model by itself but that can be added to an existing model to extend its capabilities.

For instance if one adds a module *vace_14B* on top of a model with architecture *t2v* one gets get a model with the *vace_14B* architecture. Here *vace_14B* stands for both an architecture name and a module name. The module system allows you to reuse shared weights between models.


## The Model Subtree
- *name* : name of the finetune used to select
- *architecture* : architecture Id of the base model of the finetune (see previous section)
- *finetune_source_model*: optional source model id used by the Finetune Editor when the source model id differs from *architecture*. If omitted, WanGP uses *architecture* as the source model.
- *description*: description of the finetune that will appear at the top
- *infos*: optional Markdown shown from the small information button next to the model description. Use it for model-level notes: what the finetune is, recommended resolutions or settings, known limitations, license/source notes, or what makes this finetune different from the base model.
- *prompt_infos*: optional Markdown shown from the small information button next to the prompt label. Use it for prompt-writing guidance: expected prompt syntax, examples, special tags, speaker formats, Prompt Relay syntax, or any model-specific wording rules. If your text explains how to write the prompt, prefer *prompt_infos* over *infos*.
- *URLs*: URLs of all the finetune versions (quantized / non quantized). WanGP will pick the version that is the closest to the user preferences. You will need to follow a naming convention to help WanGP identify the content of each version (see next section). WanGP supports 8 bits quantized model that have been quantized using **quanto** and Scaled FP8 models. WanGP offers a command switch to build easily such a quantized model (see below). *URLs* can contain also paths to local file to allow testing.
- *URLs2*: URLs of all the finetune versions (quantized / non quantized) of the weights used for the second phase of a model. For instance with Wan 2.2, the first phase contains the High Noise model weights and the second phase contains the Low Noise model weights. This feature can be used with other models than Wan 2.2 to combine different model weights during the same video generation.
- *text_encoder_URLs* : URLs of the text_encoder versions (quantized or not), if specified will override the default text encoder
- *VAE_URLs* : URL of a VAE (in a list), if specified will override the default VAE (supported so far only with Wan & LTX2 models)
- *configs*: optional dictionary of selectable loading configurations defined by the finetune author. It may instead contain the id of another model whose user config dictionary should be reused. The reserved *_name* entry sets the dropdown label, *_default_label* renames its automatic **Default** option, and *_infos* provides Markdown help beside the label. Every other key is a config id whose value is a dictionary that overrides properties of the enclosing *model* subtree. A config-level *name* is used as the option label in the UI; otherwise its id is displayed. Any system configurations supplied by the architecture are applied before this user configuration.
- *modules*: this a list of modules to be combined with the models referenced by the URLs. A module is a model extension that is merged with a model to expand its capabilities. Supported models so far are : *vace_14B* and *multitalk*. For instance the full Vace model is the fusion of a Wan text 2 video and the Vace module.
- *preload_URLs* : URLs of files to download no matter what (used to load quantization maps for instance)
- *loras* : URLs or file paths of LoRAs that will be applied before any LoRA selected by the user. These LoRAs will often be accelerators. For instance if you specify here a FusioniX LoRA you will be able to reduce the number of generation steps to 10.
- *loras_multipliers* : a list of float numbers or strings that defines the weight of each LoRA mentioned in *loras*. The order must match the order of *loras*. The string syntax is used if you want your LoRA multiplier to change over the steps (please check the [LoRAs guide](LORAS.md)) or if you want a multiplier to be applied on a specific High Noise phase or Low Noise phase of a Wan 2.2 model. For instance, here the multiplier will be only applied during the High Noise phase and for half of the steps of this phase the multiplier will be 1 and for the other half 1.1.
```
"loras" : [ "my_lora.safetensors"],
"loras_multipliers" : [ "1,1.1;0"]
```

- *auto_quantize*: if set to True and no quantized model URL is provided, WanGP will perform on the fly quantization if the user expects a quantized model
- *visible* : by default assumed to be true. If set to false the model will no longer be visible. This can be useful if you create a finetune to override a default model and hide it.
- *image_outputs* : turn any model that generates a video into a model that generates images. In fact it will adapt the user interface for image generation and ask the model to generate a video with a single frame.
- *resolutions*: optional explicit list of resolutions for this model. Each entry is a label/value pair such as `["1024x2048 (1:2)", "1024x2048"]`. In the Finetune Editor you can enter one `WxH` value per line and WanGP will generate these labels automatically. If *resolutions_categories* is also set, both lists are added together. Displayed and stored resolution values are adjusted to the model `vae_block_size`; any ratio text already present in the label is kept as written.
- *resolutions_categories*: optional list of resolution category conditions. When set, WanGP builds the resolution list from the global built-in/custom resolutions that match these categories, overriding the global 3K/4K+ availability switch for this model. Conditions use the same operators as CUDA architecture rules: exact values such as *720p*, *2k*, or *4096p*, comparisons such as *>=720* or *<=1080*, AND with *&*, and OR with *+*. Multiple list entries are also OR conditions. Supported aliases include *2k* for *1440p* and *4k* for *2160p*.
- *text_prompt_enhancer_instructions* : this allows you override the system prompt used by the Prompt Enhancer if only a Prompt about a text is requested
- *video_prompt_enhancer_instructions* : this allows you override the system prompt used by the Prompt Enhancer when generating a Video with this finetune
- *image_prompt_enhancer_instructions* : this allows you override the system prompt used by the Prompt Enhancer when generating an Image with this finetune
- *text_prompt_enhancer_max_tokens*: override for the maximum number of tokens generated by the prompt enhancer if only a Prompt about a text is requested (default 256)
- *video_prompt_enhancer_max_tokens*: override for the maximum number of tokens generated by the prompt enhancer when generating a Video (default 256)
- *image_prompt_enhancer_max_tokens*: override for the maximum number of tokens generated by the prompt enhancer when generating an Image (default 256)

Example:
```
"model": {
  "name": "My Cinematic Finetune",
  "architecture": "ltx2",
  "description": "A stylized LTX2 finetune for noir dialogue scenes.",
  "resolutions": [["1024x2048 (1:2)", "1024x2048"]],
  "infos": "## Model Notes\nWorks best at 1216x704 with 30 to 40 steps. Use moderate guidance for stable faces.",
  "prompt_infos": "## Prompt Format\nWrite one paragraph per shot. Put spoken words in double quotes. Keep character names consistent across the whole prompt."
}
```

### Selectable Model Loading Configurations

Use the optional *configs* dictionary when several component combinations should remain under one finetune entry. It adds one user configuration dropdown after any configuration dropdowns supplied by the architecture. The architecture can supply up to three such dropdowns, so the user dropdown always occupies the fourth position in the config row.

Every populated config dictionary produces a dropdown whose first choice has an empty value and is labeled **Default**. Set *_default_label* in the dictionary to use a different label for that choice. Set *_name* to choose the dropdown label; if it is missing or empty, the label is **config**. Add optional Markdown in *_infos* to explain how to choose between its configurations: an **i** help icon appears beside the dropdown label, like the prompt and upsampler help. These three keys are reserved and are not displayed as selectable configurations. They work in both the architecture's *system_configs*, *system_configs2*, *system_configs3* dictionaries and the finetune's *configs*. Do not add an explicit empty/default config.

The selections are stored in the existing generation setting named *config*, in dropdown order and separated by commas. Empty positions are preserved when needed, while trailing empty positions are omitted. Because the user config is the fourth dropdown, selecting `alternate_text_encoder` there is stored as `,,,alternate_text_encoder`. The UI handles this representation automatically. Config ids must not contain commas, and a saved id that no longer exists resolves to that dropdown's Default option. An all-Default selection is stored as an empty string.

Config values are shallow overrides of the enclosing *model* dictionary and are applied only when the model is loaded. System configurations supplied by the architecture are applied first, in their displayed order, and the selected user config is applied last. The user config therefore wins when it overrides the same property. Properties omitted from all selected configs continue to use the enclosing model value or the value inherited from its architecture. Changing any config dropdown reloads the current model, and all selected configs are recorded in the generated media information.

The complete user config dictionary can be inherited from another model definition:

```json
"configs": "t2v"
```

This reuses the *configs* dictionary declared by *t2v*, while a selected config still overrides the enclosing model in which the reference appears.

This example keeps the same transformer while allowing one alternate component set to be selected for the finetune. The standard architecture components come from the automatic default option, so no explicit standard config is needed:

```json
{
  "model": {
    "name": "My LTX2 Finetune",
    "architecture": "ltx2_22B",
    "description": "One transformer with selectable text encoder and VAE configurations.",
    "URLs": [
      "https://huggingface.co/your-account/your-repo/resolve/main/my_ltx2_finetune_bf16.safetensors",
      "https://huggingface.co/your-account/your-repo/resolve/main/my_ltx2_finetune_quanto_bf16_int8.safetensors"
    ],
    "configs": {
      "_name": "Finetune Components",
      "_default_label": "Architecture Default",
      "_infos": "Use **Architecture Default** for the standard components. Choose **Alternate Text Encoder** to change prompt interpretation, or **Alternate VAE** to change image decoding.",
      "alternate_text_encoder": {
        "name": "Alternate Text Encoder",
        "text_encoder_URLs": [
          "https://huggingface.co/your-account/your-repo/resolve/main/alternate_text_encoder_bf16.safetensors",
          "https://huggingface.co/your-account/your-repo/resolve/main/alternate_text_encoder_quanto_bf16_int8.safetensors"
        ]
      },
      "alternate_vae": {
        "name": "Alternate VAE",
        "VAE_URLs": [
          "https://huggingface.co/your-account/your-repo/resolve/main/alternate_vae.safetensors"
        ]
      }
    }
  },
  "prompt": "A cinematic scene used to compare the available model configurations."
}
```

The alternate text encoder must be compatible with the architecture and its tokenizer. If it needs a different component folder, also override *text_encoder_folder* and make sure that folder contains the required tokenizer files. The alternate VAE must likewise be supported by the selected architecture.

In order to favor reusability the properties of *URLs*, *modules*, *loRAs* and  *preload_URLs* can contain instead of a list of URLs a single text which corresponds to the id of a finetune or default model to reuse. Instead of:
```
    "URLs": [
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_mbf16.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_quanto_mbf16_int8.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_quanto_mfp16_int8.safetensors"
    ],
    "URLs2": [
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_mbf16.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_quanto_mbf16_int8.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_quanto_mfp16_int8.safetensors"
    ],
```
 You can write:
```
 "URLs":  "t2v_2_2",
 "URLs2":  "t2v_2_2",
```


Example of **model** subtree
```
        "model":
        {
                "name": "Wan text2video FusioniX 14B",
                "architecture" : "t2v",
                "description": "A powerful merged text-to-video model based on the original WAN 2.1 T2V model, enhanced using multiple open-source components and LoRAs to boost motion realism, temporal consistency, and expressive detail. multiple open-source models and LoRAs to boost temporal quality, expressiveness, and motion realism.",
                "URLs": [
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_fp16.safetensors",
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_fp16_int8.safetensors",
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_bf16_int8.safetensors"
                ],
        "preload_URLs": [
        ],
                "auto_quantize": true
        },
```

## Finetune Model Naming Convention
If a model is not quantized, it is assumed to be mostly 16 bits (with maybe a few 32 bits weights), so *bf16* or *fp16* should appear somewhere in the name. If you need examples just look at the **ckpts** subfolder, the naming convention for the base models is the same.

If a model is quantized the term *quanto* should also be included since WanGP supports for the moment only *quanto* quantized model, most specically you should replace *fp16* by *quanto_fp16_int8* or *bf6* by *quanto_bf16_int8*.

Please note it is important than *bf16", "fp16* and *quanto* are all in lower cases letters.

## Creating a Quanto Quantized file
If you launch the app with the *--save-quantized* switch, WanGP will create a quantized file in the **ckpts** subfolder just after the model has been loaded. Please note that the model will *bf16* or *fp16* quantized depending on what you chose in the configuration menu.

1) Make sure that in the finetune definition json file there is only a URL or filepath that points to the non quantized model
2) Launch WanGP *python wgp.py --save-quantized*
3) In the configuration menu *Transformer Data Type* property choose either *BF16* of *FP16*
4) Launch a video generation (settings used do not matter). As soon as the model is loaded, a new quantized model will be created in the **ckpts** subfolder if it doesn't already exist.
5) WanGP will update automatically the finetune definition file with the local path of the newly created quantized file (the list "URLs" will have an extra value such as *"ckpts/finetune_quanto_fp16_int8.safetensors"*
6) Remove *--save-quantized*, restart WanGP and select *Scaled Int8 Quantization* in the *Transformer Model Quantization* property
7) Launch a new generation and verify in the terminal window that the right quantized model is loaded
8) In order to share the finetune definition file you will need to store the fine model weights in the cloud. You can upload them for instance on *Huggingface*. You can now replace in the finetune definition file the local path by a URL (on Huggingface to get the URL of the model file click *Copy download link* when accessing the model properties)

You need to create a quantized model specifically for *bf16* or *fp16* as they can not converted on the fly. However there is no need for a non quantized model as they can be converted on the fly while being loaded.

Wan models supports both *fp16* and *bf16* data types albeit *fp16* delivers in theory better quality. On the contrary Hunyuan and LTXV supports only *bf16*.

## Using the Finetune Creator / Editor

The model toolbar contains a finetune tool. When the currently selected model is a base model, the tool opens the **Finetune Creator**. When the selected model is already a finetune, the tool opens the **Finetune Editor**. Both modes use the same shortcut.

In **Creator** mode:
1) Select the base model you want to derive from.
2) Open the finetune tool from the model toolbar.
3) Choose **Using Current Model** to create a new finetune from the selected model, or **By importing a File** to import an existing finetune JSON.
4) Fill **Id**, **Name**, and **Description**. If **auto** is enabled, WanGP generates the id from the source model and the finetune name.
5) In the **URLs** tab, set checkpoint files or URLs. Use the folder icon next to each field to pick local files from the server machine. Multiple checkpoint fields accept one entry per line.
6) In the **LoRAs** tab, optionally add always-loaded LoRAs and matching multipliers. Use one LoRA per line; multipliers are ordered the same way as the LoRA list.
7) In the **Resolutions** tab, optionally add resolution category conditions, one per line, or custom resolutions, one `WxH` value per line. Category lines use OR between lines, for example `720p`, `1080p`, and `2k`. A range can be written on one line, for example `>=720&<=1440`. Custom resolution lines such as `1024x2048` are saved with an automatically generated label such as `1024x2048 (1:2)`. WanGP validates these fields before saving.
8) In the **Help** tab, optionally customize **Model Infos** and **Prompt Help** markdown.
9) In the **Prompt Enhancer** tab, optionally override the source model prompt enhancer instructions. If an eye icon is shown, it copies the source model system prompt into the field.
10) Enable **Use Current Model Settings as Default Settings** if you want the current UI settings to become the finetune defaults.
11) Click **Create** to create and switch to the new finetune, or **Create & New** to create it and immediately start another finetune.

In **Editor** mode:
1) Select the finetune in the model hierarchy. Finetunes are marked with a `*` in the third level of the selector.
2) Open the finetune tool from the toolbar.
3) Edit the same fields as in Creator mode. Changing the **Id** renames the finetune JSON file; WanGP also renames the matching settings file when one exists.
4) Click **Save** to update the finetune, **Export** to download/share its JSON definition, or **Delete** to remove it. Delete shows a confirmation row in place of the editor action buttons.

After creation, import, save, or delete, WanGP refreshes the model list automatically.

---

> Applies to: Creating and managing custom model definitions and weights. Model-definition fields and conversion commands concern model setup; per-generation settings are specified separately.
