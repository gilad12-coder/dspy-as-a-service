/**
 * Terms of Service copy for the hosted Skynet service.
 *
 * Original prose tailored to Skynet's real mechanics (prepaid credits via
 * Stripe, bring-your-own-key runs, third-party LLM providers, AGPL software vs.
 * hosted service). This is a launch-ready draft, not legal advice — have
 * counsel review it before relying on it. Operator-specific values (entity,
 * governing law and contact emails) live in legal-config.ts.
 */

import { LEGAL_CONFIG as C } from "./legal-config";
import type { LegalDocument } from "./types";

export const TERMS_OF_SERVICE: LegalDocument = {
  title: "Terms of Service",
  intro:
    `These Terms of Service ("Terms") govern your access to and use of ${C.serviceName}, ` +
    `the prompt-optimization service available at ${C.websiteLabel} (the "Service"), operated by ` +
    `${C.legalEntity} ("${C.serviceName}", "we", "us", or "our"). By creating an account or using ` +
    `the Service, you agree to these Terms. If you do not agree, do not use the Service.`,
  sections: [
    {
      heading: "The Service",
      blocks: [
        {
          kind: "paragraph",
          text:
            `${C.serviceName} is a platform for optimizing and evaluating prompts for large language ` +
            `models ("LLMs") using automated techniques. You provide datasets, prompts, and ` +
            `evaluation criteria; the Service runs optimization jobs against LLMs and returns ` +
            `improved prompts, metrics, and results.`,
        },
        {
          kind: "paragraph",
          text:
            `The underlying ${C.serviceName} software is open source under the GNU Affero General ` +
            `Public License v3 (AGPL v3). These Terms govern the hosted Service we operate, which is ` +
            `separate from, and additional to, that software license. If you run your own copy of the ` +
            `software, these Terms do not apply to your deployment.`,
        },
      ],
    },
    {
      heading: "Eligibility and accounts",
      blocks: [
        {
          kind: "paragraph",
          text:
            `You must be at least 18 years old, or the age of majority in your jurisdiction, to use ` +
            `the Service. You agree to provide accurate registration information and to keep it ` +
            `current.`,
        },
        {
          kind: "paragraph",
          text:
            `You may sign in with an email and password or with a supported third-party identity ` +
            `provider (for example, Google or GitHub). Your use of a third-party provider is subject ` +
            `to that provider's terms.`,
        },
        {
          kind: "paragraph",
          text:
            `You are responsible for all activity under your account and for safeguarding your ` +
            `credentials, including passwords, passkeys, two-factor codes, recovery codes, and ` +
            `personal access tokens. Notify us promptly if you suspect unauthorized use of your ` +
            `account.`,
        },
      ],
    },
    {
      heading: "Acceptable use",
      blocks: [
        { kind: "paragraph", text: "You agree not to:" },
        {
          kind: "list",
          items: [
            "Break the law or infringe the intellectual-property, privacy, or other rights of others.",
            "Upload content you do not have the rights to use, or that contains other people's personal data without a lawful basis.",
            "Access another user's account or data, or attempt to bypass access controls, usage limits, quotas, or billing.",
            "Interfere with, disrupt, overload, or probe the Service or its infrastructure, except as expressly permitted by the open-source license for the software itself.",
            "Use the Service to generate or distribute unlawful, harmful, deceptive, or abusive content, or to violate the acceptable-use policies of the LLM providers reached through the Service.",
            "Resell, sublicense, or provide the hosted Service to third parties except as expressly permitted by us.",
            "Use automated means to extract data from the Service beyond the interfaces we provide.",
          ],
        },
        {
          kind: "paragraph",
          text:
            "We may investigate suspected violations and may suspend or terminate accounts that " +
            "violate these rules or create risk for us, other users, or third parties.",
        },
      ],
    },
    {
      heading: "Your content",
      blocks: [
        {
          kind: "paragraph",
          text:
            `You retain all rights to the datasets, prompts, evaluation code, and other materials you ` +
            `submit ("Your Content"), and to the outputs the Service generates for you. We do not ` +
            `claim ownership of Your Content.`,
        },
        {
          kind: "paragraph",
          text:
            `You grant us a limited, non-exclusive license to host, store, process, transmit, and ` +
            `display Your Content solely to operate and provide the Service to you. This includes ` +
            `sending Your Content to the LLM, payment, and infrastructure providers described in our ` +
            `Privacy Policy, as needed to run your jobs.`,
        },
        {
          kind: "paragraph",
          text:
            "You are responsible for Your Content and represent that you have the necessary rights " +
            "and lawful bases to submit it and to have it processed as described. Do not upload " +
            "sensitive personal data unless you have a lawful basis to do so.",
        },
      ],
    },
    {
      heading: "AI outputs and third-party model providers",
      blocks: [
        {
          kind: "paragraph",
          text:
            "To perform optimization, the Service sends your prompts and dataset content to " +
            "third-party LLM providers (through OpenRouter and, where we have configured one, a " +
            "self-hosted gateway). Your use of the Service is also subject to those providers' terms " +
            "and acceptable-use policies.",
        },
        {
          kind: "paragraph",
          text:
            "AI-generated outputs may be inaccurate, incomplete, or unsuitable for your purpose. You " +
            "are responsible for reviewing and validating outputs before relying on them. We do not " +
            "warrant that outputs are correct, fit for a particular purpose, or free of third-party " +
            "rights, and you should not treat outputs as a substitute for professional advice.",
        },
      ],
    },
    {
      heading: "Bring your own key (BYOK)",
      blocks: [
        {
          kind: "paragraph",
          text:
            "If you provide your own third-party provider API key, you authorize us to store it " +
            "encrypted and to use it to run your jobs against that provider on your behalf. We store " +
            "provider keys encrypted at rest and do not display the full key back to you.",
        },
        {
          kind: "paragraph",
          text:
            "You represent that you are authorized to use any key you provide, and you remain " +
            "responsible for your provider account, its usage, its costs, and your compliance with " +
            "that provider's terms.",
        },
      ],
    },
    {
      heading: "Credits, billing, and refunds",
      blocks: [
        {
          kind: "list",
          items: [
            "The Service uses a prepaid credit model. You purchase credit packs, and credits are consumed as you run jobs. Prices and the credit cost of features are shown in the Service and may change.",
            "Payments are processed by our payment processor, Stripe. We do not receive or store your full card details. Your purchases are also subject to Stripe's terms.",
            "Except where required by law or expressly stated otherwise, credit purchases are final and non-refundable. If a charge is refunded, reversed, or disputed (including chargebacks), we may deduct the corresponding credits from your balance and may suspend access while the matter is resolved.",
            "You are responsible for any taxes associated with your purchases, other than taxes based on our net income.",
            "Free or promotional credits have no cash value and may expire or be revoked.",
          ],
        },
      ],
    },
    {
      heading: "Availability, limits, and changes to the Service",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We may modify, suspend, or discontinue any part of the Service at any time. We aim for " +
            "high availability but do not guarantee that the Service will be uninterrupted, timely, " +
            "secure, or error-free.",
        },
        {
          kind: "paragraph",
          text:
            "We may set or change usage limits (such as storage quotas and rate limits) to protect " +
            "the Service, and we may retain, cap, or automatically purge certain data (such as logs " +
            "and inactive items) as described in our Privacy Policy.",
        },
      ],
    },
    {
      heading: "Intellectual property",
      blocks: [
        {
          kind: "paragraph",
          text:
            `The hosted Service, including its design, look and feel, and the "${C.serviceName}" name ` +
            `and logo, is owned by us or our licensors. Apart from the AGPL v3 license that applies to ` +
            `the open-source ${C.serviceName} codebase, these Terms grant you no rights in our ` +
            `intellectual property beyond the right to use the Service, and you may not use our marks ` +
            `without our prior written permission.`,
        },
      ],
    },
    {
      heading: "Termination",
      blocks: [
        {
          kind: "paragraph",
          text:
            "You may stop using the Service and delete your account at any time from your account " +
            "settings. We may suspend or terminate your access if you violate these Terms, to comply " +
            "with law, or to protect the Service, other users, or third parties.",
        },
        {
          kind: "paragraph",
          text:
            "On termination, your right to use the Service ends. Deleting your account removes your " +
            "content and credentials as described in our Privacy Policy; some records, such as " +
            "financial records, are retained in anonymized form as required. Provisions that by their " +
            "nature should survive termination will survive, including payment obligations, " +
            "disclaimers, limitations of liability, indemnification, and the dispute-resolution terms.",
        },
      ],
    },
    {
      heading: "Disclaimers",
      blocks: [
        {
          kind: "paragraph",
          text:
            `THE SERVICE AND ALL OUTPUTS ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ` +
            `ANY KIND, WHETHER EXPRESS, IMPLIED, OR STATUTORY, INCLUDING IMPLIED WARRANTIES OF ` +
            `MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. We do not ` +
            `warrant that the Service will be uninterrupted, secure, or error-free, or that outputs ` +
            `will be accurate or reliable. Some jurisdictions do not allow the exclusion of certain ` +
            `warranties, so some of the above may not apply to you.`,
        },
      ],
    },
    {
      heading: "Limitation of liability",
      blocks: [
        {
          kind: "paragraph",
          text:
            `TO THE MAXIMUM EXTENT PERMITTED BY LAW, ${C.serviceName} AND ITS SUPPLIERS WILL NOT BE ` +
            `LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR EXEMPLARY DAMAGES, OR FOR ` +
            `ANY LOST PROFITS, REVENUES, DATA, OR GOODWILL, ARISING OUT OF OR RELATED TO YOUR USE OF ` +
            `THE SERVICE.`,
        },
        {
          kind: "paragraph",
          text:
            "TO THE MAXIMUM EXTENT PERMITTED BY LAW, OUR TOTAL LIABILITY FOR ALL CLAIMS RELATING TO " +
            "THE SERVICE WILL NOT EXCEED THE GREATER OF THE AMOUNTS YOU PAID US IN THE THREE MONTHS " +
            "BEFORE THE EVENT GIVING RISE TO THE LIABILITY, OR ONE HUNDRED U.S. DOLLARS (USD 100). " +
            "These limitations apply even if a remedy fails of its essential purpose. Some " +
            "jurisdictions do not allow certain limitations, so some of the above may not apply to you.",
        },
      ],
    },
    {
      heading: "Indemnification",
      blocks: [
        {
          kind: "paragraph",
          text:
            `You will defend, indemnify, and hold harmless ${C.serviceName} and its officers, ` +
            `employees, and agents from and against any claims, damages, liabilities, and expenses ` +
            `(including reasonable legal fees) arising out of or related to Your Content, your use of ` +
            `the Service, your violation of these Terms, or your violation of any law or third-party ` +
            `right.`,
        },
      ],
    },
    {
      heading: "Changes to these Terms",
      blocks: [
        {
          kind: "paragraph",
          text:
            "We may update these Terms from time to time. If we make material changes, we will take " +
            "reasonable steps to notify you, for example by email or through an in-Service notice. " +
            "Changes take effect when posted unless we state otherwise, and your continued use of the " +
            "Service after that constitutes acceptance.",
        },
      ],
    },
    {
      heading: "Governing law and disputes",
      blocks: [
        {
          kind: "paragraph",
          text:
            `These Terms are governed by the laws of ${C.governingLaw}, without regard to its ` +
            `conflict-of-law rules. To the extent permitted by law, you and ${C.serviceName} agree to ` +
            `the exclusive jurisdiction of the courts located in ${C.venue} for any dispute arising ` +
            `out of or relating to these Terms or the Service. Nothing in this section limits any ` +
            `mandatory consumer-protection rights you have under the law of your place of residence.`,
        },
      ],
    },
    {
      heading: "Miscellaneous",
      blocks: [
        {
          kind: "list",
          items: [
            "These Terms, together with our Privacy Policy, are the entire agreement between you and us regarding the Service and supersede any prior agreements on that subject.",
            "If any provision is found unenforceable, the remaining provisions stay in effect.",
            "Our failure to enforce a provision is not a waiver of it.",
            "You may not assign these Terms without our prior written consent; we may assign them to an affiliate or in connection with a merger, acquisition, or sale of assets.",
            "Nothing in these Terms creates any agency, partnership, or joint venture between you and us.",
          ],
        },
      ],
    },
  ],
};
