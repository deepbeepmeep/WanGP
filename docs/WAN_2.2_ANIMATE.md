# Wan 2.2 Animate Pipeline

## High-Level Workflow

The `Wan 2.2 Animate` pipeline is designed for advanced video manipulation, allowing users to transfer body and facial motion from a source video to a character image. This can be used to either replace a person in an existing video or animate a still character image using a pose video.

The general workflow is as follows:

1.  **Input**: The user provides a source video (for motion) and a character image (the target for the animation).
2.  **Masking**: The `Mat Anyone` tool is used to create a precise mask of the person or object in the source video from which the motion will be extracted. This mask isolates the desired motion.
3.  **Generation**: The `animate` architecture takes the source video, the character image, and the generated mask as inputs. It then generates a new video where the character from the image is animated with the motion from the source video.
4.  **Relighting (Optional)**: A "Relighting" LoRA can be applied during the generation process to adjust the lighting on the animated character, helping it blend more seamlessly into the new scene.
5.  **Output**: The final output is a video where the character has been animated or replaced according to the user's inputs.

## Pipeline Components

The `Wan 2.2 Animate` pipeline is composed of several key components that work together to generate the final video.

### Masking: Mat Anyone

The first step in the pipeline is to generate a mask for the source video. This is handled by the **Mat Anyone** tool, which is an interactive masking application integrated into `WanGP`.

-   **Implementation**: The core logic for the Mat Anyone tool is located in `preprocessing/matanyone/app.py`. This script provides a Gradio interface for users to load a video and interactively "paint" a mask onto the frames.
-   **Functionality**: The tool uses a SAM (Segment Anything Model) to generate precise masks based on user clicks. Users can add positive and negative points to refine the mask until it accurately isolates the desired person or object.
-   **Output**: The output of this stage is a black-and-white video mask that is used in the generation stage to specify the area of motion to be transferred.

### Model and Architecture

The core of the `Wan 2.2 Animate` pipeline is its custom architecture and model definition.

-   **Model Definition**: The pipeline's configuration is defined in `defaults/animate.json`. This file specifies the model's name, architecture, and the URLs for the model weights. It also preloads the "Relighting" LoRA.
-   **Architecture**: The pipeline uses a dedicated `"animate"` architecture, which is implemented in the `models/wan/animate/` directory. The key files in this directory are:
    -   `model_animate.py`: The main model definition.
    -   `motion_encoder.py`: Encodes the motion from the source video.
    -   `face_blocks.py`: Handles facial animations.
    -   `animate_utils.py`: Provides utility functions for the animation process.
-   **Model Weights**: The main model weights, `wan2.2_animate_14B_bf16.safetensors`, are downloaded from the URL specified in `defaults/animate.json`.

### Relighting

To help the animated character blend more seamlessly into the target video, the pipeline includes an optional "Relighting" feature.

-   **Implementation**: Relighting is implemented as a LoRA (Low-Rank Adaptation) that is applied to the main model during the generation process.
-   **LoRA File**: The LoRA weights are stored in `wan2.2_animate_relighting_lora.safetensors`, which is preloaded by the application as specified in `defaults/animate.json`.
-   **Functionality**: When enabled, this LoRA adjusts the lighting on the animated character to better match the lighting of the background scene, resulting in a more realistic and cohesive final video.

## Execution Flow

The end-to-end execution flow of the `Wan 2.2 Animate` pipeline is as follows:

1.  **User Input**: The user selects the `Wan 2.2 Animate` model in the `WanGP` interface. They provide a source video for motion and a character image.
2.  **Mask Generation**: The user is directed to the `Mat Anyone` tool to create a mask for the source video. They interactively generate a mask that isolates the person or object whose motion they want to transfer.
3.  **Model Selection**: The `wgp.py` script identifies the selected model and loads the `animate` architecture based on the `defaults/animate.json` definition file.
4.  **LoRA Application (Optional)**: If the "Relighting" option is enabled, the `wan2.2_animate_relighting_lora.safetensors` LoRA is loaded and applied to the main model.
5.  **Video Generation**: The `animate` architecture's `generate` function is called with the following inputs:
    -   The source video (for motion).
    -   The character image.
    -   The generated video mask.
6.  **Motion Encoding**: The `motion_encoder.py` module processes the source video to extract the motion data.
7.  **Animation**: The `model_animate.py` module uses the extracted motion data to animate the character image. Facial animations are handled by the `face_blocks.py` module.
8.  **Output**: The final video is generated and displayed to the user in the `WanGP` interface.