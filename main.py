from process_video import process_stream, process_video

def main(video_path, analyze_stream=False):
    if analyze_stream:
        process_stream()
    else:
        process_video(video_path)


if __name__ == '__main__':
    main("vid7.mp4", False)
