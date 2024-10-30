import runpod
import json
import time
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
import threading


# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = os.environ.get("COMFY_POLLING_INTERVAL_MS", 2000)
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = os.environ.get("COMFY_POLLING_MAX_RETRIES", 300)



# Create an S3 client
s3_client = boto3.client('s3')


# Define a retry strategy
retry_strategy = Retry(
    total=20,               # Number of retry attempts
    backoff_factor=1,       # Exponential backoff factor (1 second, then 2, then 4, etc.)
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
    allowed_methods=["HEAD", "GET", "OPTIONS"]  # Methods to retry on
)

# Create an adapter with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

timeout = (2, 5)  # (connect timeout, read timeout)




def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Return validated data and no error
    return {"workflow": workflow, "images": images}, None




def upload_images(images):
    """
    Upload a list of base64 encoded images or URLs to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []
    
    print(f"[handler] uploading {len(images)} images")

    for image in images:
        name = image["name"]
        image_data = image["image"]
        blob = None

        if image_data.startswith("http"):
            response = session.get(image_data, timeout=timeout)
            if response.status_code == 200:
                blob = response.content
            else:
                upload_errors.append(f"Error downloading {name}: {response.text}")
                continue
        else:    
            blob = base64.b64decode(image_data)

        # Prepare the form data
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }

        # POST request to upload the image
        response = session.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=timeout)
        if response.status_code != 200:
            upload_errors.append(f"Error uploading {name}: {response.text}")
        else:
            responses.append(f"Successfully uploaded {name}")

    if upload_errors:
        print(f"[handler] image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"[handler] image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def process_output_images(outputs): #, job_id):
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
      defaulting to "/comfyui/output" if not set.
    - It then iterates through the outputs to find the filenames of the generated images.
    - After confirming the existence of the image in the output folder, it uploads the image 
      to the bucket and returns the URL.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")

    local_images = {}

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output["images"]:
                if image["type"] == "output":
                    local_images[node_id] = os.path.join(image["subfolder"], image["filename"])

    print(f"[handler] image generation is done")

    output_images = {}

    for node_id, output_image in local_images.items():
        print(f"[handler] output image {node_id}: {output_image}")

        # expected image output folder
        local_image_path = f"{COMFY_OUTPUT_PATH}/{output_image}"

        print(f"[handler] {local_image_path}")

        try:
            if os.path.exists(local_image_path) == False:
                raise FileNotFoundError(f"the image does not exist in the output folder: {local_image_path}")

            bucket_name = os.environ.get("S3_BUCKET")
            id = uuid.uuid4()
            ext = os.path.splitext(local_image_path)[1]
            s3_key = f"{id}{ext}"
            s3_client.upload_file(local_image_path, bucket_name, s3_key)
 
            url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            output_images[node_id] = url
            
            print(f"[handler] the image was generated and uploaded to {url}")

        except NoCredentialsError:
            print("[handler] AWS credentials not available")
            return {
                "status": "error",
                "message": "AWS credentials not available",
            }
            
        except FileNotFoundError as e:
            print(f"[handler] {str(e)}")
            return {
                "status": "error",
                "message": str(e),
            }
    
    return {
        "status": "success",
        "message": output_images,
    }
    

def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    images = validated_data.get("images")

    print(f"[handler] found {len(images)} images to upload")


    # Make sure that the ComfyUI API is available
    try:
        response = session.get(f"http://{COMFY_HOST}", timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to ComfyUI: {str(e)}"}
    except Exception as e:
        return {"error": f"Unknown error: {str(e)}"}


    # Upload images if they exist
    upload_result = upload_images(images)

    if upload_result["status"] == "error":
        return upload_result

    # Queue the workflow
    try:
        data = json.dumps({"prompt": workflow}).encode("utf-8")
        response = session.post(f"http://{COMFY_HOST}/prompt", data=data, timeout=timeout)
        queued_workflow = response.json()

        prompt_id = queued_workflow["prompt_id"]
        print(f"[handler] queued workflow with ID {prompt_id}")

    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to ComfyUI: {str(e)}"}
    
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    # Poll for completion
    print(f"[handler] wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            
            response = session.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=timeout)
            response.raise_for_status()
            history = response.json()

            print(f"[handler] poll attempt {retries}")

            # Exit the loop if we have found the history
            if prompt_id in history:
                    status = history[prompt_id].get("status")
                    if status["status_str"]=="error":
                        return status.get("messages")
                    
                    if history[prompt_id].get("outputs"):
                        print(f"[handler] found history with outputs")
                        #print(history[prompt_id].get("outputs"))
                        break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to ComfyUI: {str(e)}"}

    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}

    # Get the generated image and return it as URL in an AWS bucket or as base64
    images_result = process_output_images(history[prompt_id].get("outputs")) #, job["id"])

    result = {**images_result, "refresh_worker": REFRESH_WORKER}

    return result


def read_pipe():
    with open('/tmp/comfy_pipe', 'r') as pipe:
        while True:
            line = pipe.readline()
            if line:
                print("[comfy] ", line.strip())
            else:
                break  # Exit if there’s no more data


# Start the handler only if this script is run directly
if __name__ == "__main__":
    pipe_thread = threading.Thread(target=read_pipe, daemon=True)
    pipe_thread.start()
    runpod.serverless.start({"handler": handler})
