import os
import sys
import yt_dlp

def download_video(url, output_path='.'):
    """
    Downloads a video from YouTube using yt-dlp.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"开始下载: {url}")
            ydl.download([url])
            print("\n下载完成！视频已保存到当前目录。")
    except Exception as e:
        print(f"下载过程中出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python download_video.py <YouTube视频链接>")
        # 提供一个示例链接供测试（这是一个简短的YouTube官方示例视频）
        example_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        print(f"例如: python download_video.py {example_url}")
        
        choice = input(f"\n是否要下载测试视频? (y/n): ")
        if choice.lower() == 'y':
            download_video(example_url)
    else:
        video_url = sys.argv[1]
        download_video(video_url)
