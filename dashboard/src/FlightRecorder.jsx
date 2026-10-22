
import { useEffect, useRef, useState } from "react";
import { Circle, Square, Download, Video } from "lucide-react";

export default function FlightRecorder() {
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const urlRef = useRef(null);

  const [recording, setRecording] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [error, setError] = useState("");

  function startRecording() {
    setError("");

    const canvas = document.querySelector(".viewport canvas");

    if (!canvas) {
      setError("Rocket viewport not found.");
      return;
    }

    if (!window.MediaRecorder || !canvas.captureStream) {
      setError("Your browser does not support canvas recording.");
      return;
    }

    const mimeType = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ].find((type) => MediaRecorder.isTypeSupported(type));

    if (!mimeType) {
      setError("No supported video codec found.");
      return;
    }

    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
      setVideoUrl(null);
    }

    chunksRef.current = [];

    try {
      const stream = canvas.captureStream(30);
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 8_000_000,
      });

      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeType,
        });

        const url = URL.createObjectURL(blob);
        urlRef.current = url;
        setVideoUrl(url);
        setRecording(false);

        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      };

      recorder.onerror = () => {
        setError("Recording failed.");
        setRecording(false);
      };

      recorder.start(1000);
      setRecording(true);
    } catch (err) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      setError(err.message);
      setRecording(false);
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  }

  function downloadRecording() {
    if (!videoUrl) return;

    const link = document.createElement("a");
    link.href = videoUrl;
    link.download = `ZERO_SPACE_FLIGHT_${new Date()
      .toISOString()
      .replaceAll(":", "-")}.webm`;

    link.click();
  }

  useEffect(() => {
    return () => {
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());

      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
      }
    };
  }, []);

  return (
    <div className="flight-recorder">
      <div className="recorder-heading">
        <Video size={15} />
        <span>FLIGHT RECORDER</span>
      </div>

      <div className="recorder-controls">
        {!recording ? (
          <button onClick={startRecording} className="record-button">
            <Circle size={13} fill="#ff4545" />
            START RECORDING
          </button>
        ) : (
          <button onClick={stopRecording} className="stop-button">
            <Square size={13} fill="currentColor" />
            STOP RECORDING
          </button>
        )}

        {videoUrl && !recording && (
          <button onClick={downloadRecording} className="download-button">
            <Download size={14} />
            DOWNLOAD VIDEO
          </button>
        )}
      </div>

      {recording && (
        <div className="recording-status">
          <span className="recording-dot" />
          REC ● CAPTURING 30 FPS
        </div>
      )}

      {error && <p className="recorder-error">{error}</p>}
    </div>
  );
}
