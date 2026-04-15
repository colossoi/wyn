export type Status = "loading" | "ready" | "compiling" | "error";

interface StatusBarProps {
  status: Status;
  text: string;
}

export function StatusBar({ status, text }: StatusBarProps) {
  return (
    <div className="status">
      <div className={`status-dot ${status}`} />
      <span>{text}</span>
    </div>
  );
}
