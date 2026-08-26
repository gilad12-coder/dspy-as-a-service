{{/*
Standard naming + labelling helpers for the skynet chart.
*/}}

{{- define "skynet.name" -}}
{{- default .Chart.Name .Values.global.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "skynet.fullname" -}}
{{- if .Values.global.fullnameOverride -}}
{{- .Values.global.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.global.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "skynet.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Per-component fullnames */}}
{{- define "skynet.backend.fullname" -}}
{{- printf "%s-backend" (include "skynet.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "skynet.frontend.fullname" -}}
{{- printf "%s-frontend" (include "skynet.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "skynet.postgres.fullname" -}}
{{- printf "%s-postgres" (include "skynet.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "skynet.pgbouncer.fullname" -}}
{{- printf "%s-pgbouncer" (include "skynet.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Common labels applied to all resources */}}
{{- define "skynet.labels" -}}
helm.sh/chart: {{ include "skynet.chart" . }}
{{ include "skynet.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: skynet
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "skynet.selectorLabels" -}}
app.kubernetes.io/name: {{ include "skynet.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Component-scoped selector labels */}}
{{- define "skynet.backend.selectorLabels" -}}
{{ include "skynet.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end -}}

{{- define "skynet.frontend.selectorLabels" -}}
{{ include "skynet.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end -}}

{{- define "skynet.postgres.selectorLabels" -}}
{{ include "skynet.selectorLabels" . }}
app.kubernetes.io/component: postgres
{{- end -}}

{{- define "skynet.pgbouncer.selectorLabels" -}}
{{ include "skynet.selectorLabels" . }}
app.kubernetes.io/component: pgbouncer
{{- end -}}

{{/* Service account names */}}
{{- define "skynet.backend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- printf "%s-backend" (include "skynet.fullname" .) -}}
{{- else -}}
default
{{- end -}}
{{- end -}}

{{- define "skynet.frontend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- printf "%s-frontend" (include "skynet.fullname" .) -}}
{{- else -}}
default
{{- end -}}
{{- end -}}

{{/* Image references */}}
{{- define "skynet.backend.image" -}}
{{- $reg := .Values.global.imageRegistry -}}
{{- $repo := .Values.backend.image.repository -}}
{{- $tag := default .Chart.AppVersion .Values.backend.image.tag -}}
{{- if $reg -}}
{{- printf "%s/%s:%s" $reg $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{- define "skynet.frontend.image" -}}
{{- $reg := .Values.global.imageRegistry -}}
{{- $repo := .Values.frontend.image.repository -}}
{{- $tag := default .Chart.AppVersion .Values.frontend.image.tag -}}
{{- if $reg -}}
{{- printf "%s/%s:%s" $reg $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{- define "skynet.postgres.image" -}}
{{- $reg := .Values.global.imageRegistry -}}
{{- $repo := .Values.postgres.image.repository -}}
{{- $tag := .Values.postgres.image.tag -}}
{{- if $reg -}}
{{- printf "%s/%s:%s" $reg $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{- define "skynet.pgbouncer.image" -}}
{{- $reg := .Values.global.imageRegistry -}}
{{- $repo := .Values.pgbouncer.image.repository -}}
{{- $tag := .Values.pgbouncer.image.tag -}}
{{- if $reg -}}
{{- printf "%s/%s:%s" $reg $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{/* Image pull secrets block (rendered list) */}}
{{- define "skynet.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end -}}
{{- end -}}

{{/* Backend secret name (existing wins, otherwise chart-managed) */}}
{{- define "skynet.backend.secretName" -}}
{{- if .Values.backend.secrets.existingSecret -}}
{{- .Values.backend.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "skynet.backend.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "skynet.frontend.secretName" -}}
{{- if .Values.frontend.secrets.existingSecret -}}
{{- .Values.frontend.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "skynet.frontend.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* Postgres secret name */}}
{{- define "skynet.postgres.secretName" -}}
{{- if .Values.postgres.auth.existingSecret -}}
{{- .Values.postgres.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-postgres" (include "skynet.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* DB host resolution: external if enabled, else bundled postgres service */}}
{{- define "skynet.dbHost" -}}
{{- if .Values.externalDatabase.enabled -}}
{{- .Values.externalDatabase.host -}}
{{- else -}}
{{- include "skynet.postgres.fullname" . -}}
{{- end -}}
{{- end -}}

{{- define "skynet.dbPort" -}}
{{- if .Values.externalDatabase.enabled -}}
{{- .Values.externalDatabase.port -}}
{{- else -}}
{{- .Values.postgres.service.port -}}
{{- end -}}
{{- end -}}

{{- define "skynet.backendDbHost" -}}
{{- if .Values.pgbouncer.enabled -}}
{{- include "skynet.pgbouncer.fullname" . -}}
{{- else -}}
{{- include "skynet.dbHost" . -}}
{{- end -}}
{{- end -}}

{{- define "skynet.backendDbPort" -}}
{{- if .Values.pgbouncer.enabled -}}
{{- .Values.pgbouncer.service.port -}}
{{- else -}}
{{- include "skynet.dbPort" . -}}
{{- end -}}
{{- end -}}

{{- define "skynet.dbName" -}}
{{- if .Values.externalDatabase.enabled -}}
{{- .Values.externalDatabase.database -}}
{{- else -}}
{{- .Values.postgres.auth.database -}}
{{- end -}}
{{- end -}}

{{- define "skynet.dbUser" -}}
{{- if .Values.externalDatabase.enabled -}}
{{- .Values.externalDatabase.user -}}
{{- else -}}
{{- .Values.postgres.auth.username -}}
{{- end -}}
{{- end -}}

{{/*
Browser-reachable API URL injected by the frontend. An explicit value wins;
otherwise an explicit OpenShift backend Route host is used. The localhost
fallback keeps chart defaults renderable for development but must be replaced
in production.
*/}}
{{- define "skynet.frontend.apiUrl" -}}
{{- if .Values.frontend.env.API_URL -}}
{{- .Values.frontend.env.API_URL -}}
{{- else if and .Values.openshift.routes.enabled .Values.openshift.routes.backend.host -}}
{{- if .Values.openshift.routes.backend.tls.enabled -}}https{{- else -}}http{{- end -}}://{{ .Values.openshift.routes.backend.host }}
{{- else -}}
http://localhost:8000
{{- end -}}
{{- end -}}

{{/* Server-only backend address for NextAuth provisioning calls. */}}
{{- define "skynet.frontend.internalApiUrl" -}}
{{- if .Values.frontend.env.BACKEND_INTERNAL_URL -}}
{{- .Values.frontend.env.BACKEND_INTERNAL_URL -}}
{{- else -}}
{{- printf "http://%s:%d" (include "skynet.backend.fullname" .) (int .Values.backend.service.port) -}}
{{- end -}}
{{- end -}}

{{/*
Render the spec.tls block for an OpenShift Route from a tls dict that has:
  enabled, termination, insecureEdgeTerminationPolicy,
  certificate, key, caCertificate, destinationCACertificate

passthrough does not accept certificate/key/caCertificate/destinationCACertificate.
edge accepts certificate/key/caCertificate.
reencrypt accepts all four.
*/}}
{{- define "skynet.route.tls" -}}
{{- $tls := . -}}
{{- if $tls.enabled }}
tls:
  termination: {{ $tls.termination }}
  insecureEdgeTerminationPolicy: {{ $tls.insecureEdgeTerminationPolicy }}
  {{- if ne $tls.termination "passthrough" }}
  {{- with $tls.certificate }}
  certificate: |-
    {{- nindent 4 . }}
  {{- end }}
  {{- with $tls.key }}
  key: |-
    {{- nindent 4 . }}
  {{- end }}
  {{- with $tls.caCertificate }}
  caCertificate: |-
    {{- nindent 4 . }}
  {{- end }}
  {{- end }}
  {{- if eq $tls.termination "reencrypt" }}
  {{- with $tls.destinationCACertificate }}
  destinationCACertificate: |-
    {{- nindent 4 . }}
  {{- end }}
  {{- end }}
{{- end }}
{{- end -}}
