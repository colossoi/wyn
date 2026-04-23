// Landing page shown at `/` when no session is present.
//
// Keep copy sparse; the playground is the product, not the marketing site.

export function Landing() {
  return (
    <main className="landing">
      <section className="landing-hero">
        <h1>Wyn Playground</h1>
        <p className="landing-tagline">
          Write shaders in Wyn, compile to WGSL in your browser, and share
          them with a link.
        </p>
        <a href="/auth/github" className="landing-cta">
          Sign in with GitHub to save shaders
        </a>
        <p className="landing-note">
          Already have a shader link? Open it directly — viewing is public, no
          sign-in required.
        </p>
      </section>
    </main>
  );
}
