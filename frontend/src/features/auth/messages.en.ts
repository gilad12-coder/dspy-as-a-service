// English UI strings for the auth slice. Edit directly; keys missing here fall
// back to the Hebrew slice via msg(), so partial translations are safe.

import type { authMessages } from "./messages";

export const authMessagesEn: Partial<Record<keyof typeof authMessages, string>> = {
  "auth.login.error": "Sign-in failed",
  "auth.login.loading": "Signing in…",
  "auth.login.form_aria": "Sign-in form",
  "auth.login.meta_description": "Sign in to Skynet, a prompt-optimization platform",
  "auth.login.email": "Email",
  "auth.login.email_placeholder": "you@example.com",
  "auth.login.password": "Password",
  "auth.login.password_placeholder": "Your password",
  "auth.login.signin_submit": "Sign in",
  "auth.login.with_google": "Continue with Google",
  "auth.login.with_github": "Continue with GitHub",
  "auth.login.divider": "or",
  "auth.login.invalid_credentials": "Incorrect email or password",
  "auth.login.tab_signin": "Sign in",
  "auth.login.tab_signup": "Sign up",
  "auth.login.name": "Display name",
  "auth.login.name_placeholder": "What should we call you? (optional)",
  "auth.login.password_hint": "At least 8 characters",
  "auth.login.signup_submit": "Create account",
  "auth.login.register_failed": "Account creation failed. Please try again",
};
