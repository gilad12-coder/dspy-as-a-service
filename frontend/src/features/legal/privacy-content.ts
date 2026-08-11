/**
 * Privacy Policy copy for the hosted Skynet service.
 *
 * Written to match Skynet's real data flows: account/auth data, prompts and
 * datasets sent to third-party LLM providers via OpenRouter, Stripe billing,
 * encrypted-at-rest BYOK keys, operational telemetry, and the retention windows
 * enforced in the backend. This is a launch-ready draft, not legal advice — have
 * counsel review it and fill the placeholders in legal-config.ts before relying
 * on it.
 */

import { LEGAL_CONFIG as C } from "./legal-config";
import type { LegalDocument } from "./types";

export const PRIVACY_POLICY: LegalDocument = {
  title: "Privacy Policy",
  intro:
    `This Privacy Policy explains how ${C.legalEntity} ("${C.serviceName}", "we", "us", or "our") ` +
    `collects, uses, and shares personal information when you use ${C.serviceName}, the ` +
    `prompt-optimization service at ${C.websiteLabel} (the "Service"). It also describes the ` +
    `choices and rights you have. For the purposes of the GDPR, we are the controller of the ` +
    `personal information described here, except where we act as a processor for content you submit.`,
  sections: [
    {
      heading: "Information we collect",
      blocks: [
        { kind: "subheading", text: "Information you provide" },
        {
          kind: "list",
          items: [
            "Account information: your email address, display name, and password. If you sign in with a third-party provider such as Google or GitHub, we receive basic profile information from that provider.",
            "Authentication data: passkey/WebAuthn credentials, two-factor settings, recovery codes, and personal access tokens you create. Passwords and recovery codes are stored only as cryptographic hashes; we cannot read them.",
            "Content you submit: the datasets, prompts, evaluation code, module configurations, and related materials you upload or create to run optimization jobs, together with the results and metrics we generate for you.",
            "Bring-your-own-key (BYOK) credentials: if you choose to provide your own LLM provider API key, we store it encrypted at rest and use it to run your jobs. We do not display the full key back to you.",
            "Communications: messages you send us for support or other inquiries.",
          ],
        },
        { kind: "subheading", text: "Information collected automatically" },
        {
          kind: "list",
          items: [
            "Billing and transaction data: your credit balance, purchase history, and the amount of a purchase. Card payments are processed by Stripe; we receive confirmation and limited metadata (such as the last four digits and card brand) but never your full card number.",
            "Usage and job telemetry: records of the jobs you run, credits consumed, timestamps, feature usage, and error and performance logs used to operate, secure, and debug the Service.",
            "Device and connection data: IP address, browser and device type, and similar technical information, including data used for rate limiting and abuse prevention.",
            "Cookies: we use strictly necessary cookies to keep you signed in and to keep the Service secure. See “Cookies” below.",
          ],
        },
        {
          kind: "paragraph",
          text:
            "If you use the optional voice-input feature, the audio you record is sent to our speech " +
            "provider (Groq) to transcribe it into text; we do not retain the audio after " +
            "transcription.",
        },
      ],
    },
    {
      heading: "How your content reaches LLM providers",
      blocks: [
        {
          kind: "paragraph",
          text:
            "The core function of the Service is to optimize prompts against large language models. " +
            "To do this, we send the prompts and dataset content involved in your jobs to third-party " +
            "LLM providers, reached through OpenRouter and, where configured, a self-hosted gateway. " +
            "Those providers process the content to return completions, which we use to produce your " +
            "results.",
        },
        {
          kind: "paragraph",
          text:
            "We do not use your content to train our own models, and we do not sell your content. " +
            "The LLM providers' handling of the content they receive is governed by their own terms " +
            "and privacy policies. If you use BYOK, your jobs run against the provider tied to your " +
            "own key.",
        },
      ],
    },
    {
      heading: "How we use information",
      blocks: [
        {
          kind: "paragraph",
          text: "We use personal information to:",
        },
        {
          kind: "list",
          items: [
            "Provide, maintain, and improve the Service, including running your optimization jobs and returning results (legal basis: performance of a contract).",
            "Process payments, manage credit balances, and prevent payment fraud (legal basis: performance of a contract and legitimate interests).",
            "Authenticate you, secure accounts, and enforce usage limits and our Terms, including rate limiting and abuse prevention (legal basis: legitimate interests and legal obligation).",
            "Communicate with you about the Service, including security notices, transactional messages, and support (legal basis: performance of a contract and legitimate interests).",
            "Monitor, debug, and analyze the Service to keep it reliable and secure (legal basis: legitimate interests).",
            "Comply with legal obligations and respond to lawful requests (legal basis: legal obligation).",
          ],
        },
        {
          kind: "paragraph",
          text:
            "Where we rely on legitimate interests, we balance them against your rights. Where the " +
            "law requires consent, such as for any non-essential cookies, we ask for it.",
        },
      ],
    },
    {
      heading: "How we share information",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We do not sell your personal information. We share it only in these circumstances, with " +
            "service providers who process it on our behalf under appropriate contracts:",
        },
        {
          kind: "list",
          items: [
            "LLM providers (via OpenRouter, and a self-hosted gateway where configured) to run your optimization jobs.",
            "Stripe, to process payments and manage billing.",
            "Speech transcription (Groq) when you use voice input.",
            "Hosting and infrastructure providers that run our application, database, and email delivery.",
            "Professional advisers, and authorities or other parties, where necessary to comply with law, enforce our Terms, or protect the rights, property, or safety of our users or others.",
            "A successor entity in connection with a merger, acquisition, financing, or sale of assets, subject to this Policy.",
          ],
        },
        {
          kind: "paragraph",
          text:
            "We may also share aggregated or de-identified information that cannot reasonably be used " +
            "to identify you.",
        },
      ],
    },
    {
      heading: "International transfers",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We and our service providers may process your information in countries other than your " +
            "own, including the United States. Where we transfer personal information out of the EEA, " +
            "the UK, or Switzerland, we rely on appropriate safeguards such as the European " +
            "Commission's Standard Contractual Clauses, or another lawful transfer mechanism.",
        },
      ],
    },
    {
      heading: "Data retention",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We keep personal information for as long as your account is active and as needed to " +
            "provide the Service. Specific practices include:",
        },
        {
          kind: "list",
          items: [
            "Account and content data are retained until you delete the item or your account, subject to short operational delays.",
            "Some conversational and job data is automatically purged on a rolling basis, and logs are capped and rotated, so older operational data is removed over time.",
            "Billing and transaction records are retained as required for accounting, tax, and legal purposes, in de-identified form where possible after your account is closed.",
            "Backups are retained for a limited period and then overwritten on a rolling basis.",
          ],
        },
      ],
    },
    {
      heading: "Your rights and choices",
      blocks: [
        {
          kind: "paragraph",
          text:
            "Depending on where you live, you may have some or all of the following rights: to " +
            "access the personal information we hold about you; to correct inaccurate information; to " +
            "delete it; to restrict or object to certain processing; to data portability; and to " +
            "withdraw consent where processing is based on consent. If you are in the EEA or the UK, " +
            "these are your GDPR rights. If you are a California resident, you have rights under the " +
            "CCPA/CPRA, including the right to know, delete, and correct, and the right not to be " +
            "discriminated against for exercising them; we do not sell or share your personal " +
            "information as those terms are defined under California law.",
        },
        {
          kind: "paragraph",
          text:
            `You can exercise many of these rights directly in the Service: you can update your ` +
            `profile, export your data, and delete your account from your account settings. For any ` +
            `other request, contact us at ${C.privacyEmail}. We will respond as required by law, and ` +
            `you have the right to lodge a complaint with your local data protection authority.`,
        },
        {
          kind: "paragraph",
          text:
            "Please note that deleting your account removes your data from the Service, but it does " +
            "not by itself delete records held by independent third parties such as your payment " +
            "processor or, for BYOK, your own LLM provider. Those are governed by their own policies.",
        },
      ],
    },
    {
      heading: "Cookies",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We use strictly necessary cookies to keep you signed in, to secure the Service, and to " +
            "remember basic preferences. Because these cookies are essential to provide the Service, " +
            "they do not require consent. We do not use advertising cookies. You can block cookies in " +
            "your browser, but the Service may not function correctly without the essential ones.",
        },
      ],
    },
    {
      heading: "Security",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We use technical and organizational measures to protect personal information, including " +
            "encryption in transit, encryption at rest for sensitive credentials such as BYOK keys, " +
            "hashing of passwords and recovery codes, optional two-factor authentication and " +
            "passkeys, and rate limiting against abuse. No method of transmission or storage is " +
            "completely secure, so we cannot guarantee absolute security.",
        },
      ],
    },
    {
      heading: "Children's privacy",
      blocks: [
        {
          kind: "paragraph",
          text:
            "The Service is not directed to children, and we do not knowingly collect personal " +
            "information from anyone under 16. If you believe a child has provided us personal " +
            `information, contact us at ${C.privacyEmail} and we will take appropriate steps to ` +
            "delete it.",
        },
      ],
    },
    {
      heading: "Changes to this Policy",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We may update this Privacy Policy from time to time. If we make material changes, we " +
            "will take reasonable steps to notify you, for example by email or through an in-Service " +
            "notice. The “last updated” date below indicates when this Policy was last revised.",
        },
      ],
    },
    {
      heading: "Contact us",
      blocks: [
        {
          kind: "paragraph",
          text:
            `If you have questions about this Policy or how we handle your information, contact us at ` +
            `${C.privacyEmail}. The Service is operated by ${C.legalEntity}, ${C.address}.`,
        },
      ],
    },
  ],
};
