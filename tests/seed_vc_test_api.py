import argparse
import requests
import os

def test_api(source_path, target_path, output_path, host="http://127.0.0.0:11452"):
    url = f"{host}/convert"
    
    if not os.path.exists(source_path):
        print(f"Error: Source file not found: {source_path}")
        return
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return

    print(f"Sending request to {url}...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    # 准备文件
    files = {
        'source': open(source_path, 'rb'),
        'target': open(target_path, 'rb')
    }

    # 准备参数 (这里列出了所有可用参数，你可以根据需要调整)
    data = {
        'diffusion_steps': 30,
        'length_adjust': 1.0,
        'intelligibility_cfg_rate': 0.7,
        'similarity_cfg_rate': 0.7,
        'top_p': 0.9,
        'temperature': 1.0,
        'repetition_penalty': 1.0,
        'convert_style': False,
        'anonymization_only': False
    }

    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Success! Converted audio saved to: {output_path}")
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {url}. Is the service running?")
    finally:
        # 关闭文件句柄
        files['source'].close()
        files['target'].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Seed-VC v2.0 API")
    parser.add_argument("--source", type=str, required=True, help="Path to source audio file")
    parser.add_argument("--target", type=str, required=True, help="Path to target/reference audio file")
    parser.add_argument("--output", type=str, default="api_output.wav", help="Path to save the output audio")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11452", help="API server URL")
    
    args = parser.parse_args()
    test_api(args.source, args.target, args.output, args.host)
