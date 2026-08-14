"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { CircleNotch, Globe, Lock, User, UserPlus, Users, X } from "@/shared/ui/icons";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { Input } from "@/shared/ui/primitives/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/primitives/select";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { CopyButton } from "@/shared/ui/copy-button";
import { SettingsRow } from "@/shared/ui/settings-row";
import {
  addDatasetShareMember,
  getDatasetSharing,
  putDatasetSharing,
  removeDatasetShareMember,
  searchUsers,
  transferDatasetOwnership,
  updateDatasetShareMember,
  type DatasetSharingState,
  type GeneralAccess,
  type LinkRole,
  type MemberRole,
  type ShareRole,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { sessionIdentity } from "@/shared/lib/session-identity";

const ROLE_OPTIONS: MemberRole[] = ["viewer", "editor"];

// Sentinel value for the per-member role dropdown's "transfer ownership" item
// (not a real role — selecting it opens the transfer confirmation instead).
const TRANSFER_VALUE = "__transfer__";

/** Localised label for a member tier role. */
function roleLabel(role: ShareRole): string {
  if (role === "editor") return msg("share.role.editor");
  if (role === "owner") return msg("share.role.owner");
  return msg("share.role.viewer");
}

/** Localised one-line description of what a member tier role grants. */
function roleDesc(role: ShareRole): string {
  if (role === "editor") return msg("share.role.editor_desc");
  if (role === "owner") return msg("share.role.owner_desc");
  return "";
}

/**
 * Drive-style sharing modal for a library dataset. Mirrors the optimization
 * :func:`ShareDialog` — a People section (invite by username + per-member role)
 * over a General-access section (Restricted vs Anyone-with-link) plus a copy-link
 * row — minus the optimization-only Explore visibility axis. The copy link points
 * at ``/datasets/share/{token}``, redeemed by :mod:`app/datasets/share/[token]`.
 */
export function DatasetShareDialog({ datasetId }: { datasetId: string }) {
  const { data: session } = useSession();
  const me = sessionIdentity(session);
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<DatasetSharingState | null>(null);
  const [savingAccess, setSavingAccess] = useState(false);
  const [transferTarget, setTransferTarget] = useState<string | null>(null);
  const [transferring, setTransferring] = useState(false);
  // Split the confirm copy around the {name} token so the target username can
  // render emphasized below in an isolated <bdi>, matching the optimization dialog.
  // Resolved per-render (not at module scope) so it can't capture a raw key before
  // the message shim has loaded.
  const [transferBodyBefore, transferBodyAfter] = msg("share.transfer.confirm_body").split(
    "{name}",
  );

  const isOwner = !!state?.owner && state.owner.toLowerCase() === me;

  const shareUrl =
    state?.token && typeof window !== "undefined"
      ? `${window.location.origin}/datasets/share/${state.token}`
      : null;

  const accessCount = state ? (state.owner ? 1 : 0) + state.members.length : 0;

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (next && state === null) {
      getDatasetSharing(datasetId)
        .then(setState)
        .catch((err) => toast.error(err instanceof Error ? err.message : msg("share.error")));
    }
  };

  const handleAccessChange = async (value: GeneralAccess) => {
    setSavingAccess(true);
    try {
      setState(await putDatasetSharing(datasetId, { general_access: value }));
      toast.success(msg("share.access_updated"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    } finally {
      setSavingAccess(false);
    }
  };

  // The tier an "anyone with the link" link grants signed-in visitors.
  const handleLinkRoleChange = async (role: LinkRole) => {
    setSavingAccess(true);
    try {
      setState(
        await putDatasetSharing(datasetId, { general_access: "anyone", general_role: role }),
      );
      toast.success(msg("share.access_updated"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    } finally {
      setSavingAccess(false);
    }
  };

  const handleRoleChange = async (username: string, role: MemberRole) => {
    try {
      setState(await updateDatasetShareMember(datasetId, username, { role }));
      toast.success(msg("share.member_updated"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    }
  };

  // Transfer demotes the caller to editor server-side; once it lands they can no
  // longer manage — close the dialog and drop cached state so a re-open refetches.
  const handleTransfer = async () => {
    if (!transferTarget) return;
    setTransferring(true);
    try {
      // Invite-as-owner: when the target isn't a grantee yet (picked straight
      // from the invite row), add them first — the backend only hands ownership
      // to an existing member.
      const alreadyMember = state?.members.some(
        (m) => m.username.toLowerCase() === transferTarget.toLowerCase(),
      );
      if (!alreadyMember) {
        await addDatasetShareMember(datasetId, { username: transferTarget, role: "editor" });
      }
      await transferDatasetOwnership(datasetId, transferTarget);
      toast.success(msg("share.transfer.success", { name: transferTarget }));
      setTransferTarget(null);
      setOpen(false);
      setState(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    } finally {
      setTransferring(false);
    }
  };

  const handleRemove = async (username: string) => {
    try {
      setState(await removeDatasetShareMember(datasetId, username));
      toast.success(msg("share.member_removed"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    }
  };

  const handleInvite = async (username: string, role: MemberRole) => {
    setState(await addDatasetShareMember(datasetId, { username, role }));
    toast.success(msg("share.member_added"));
  };

  return (
    <>
      <TooltipButton tooltip={msg("share.button")}>
        <Button
          variant="ghost"
          size="icon-sm"
          className="size-[44px] text-muted-foreground hover:text-foreground lg:size-8"
          onClick={() => handleOpenChange(true)}
          aria-label={msg("share.button")}
        >
          <Users className="size-4" />
        </Button>
      </TooltipButton>

      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent
          className="w-[min(32rem,92vw)] max-w-[min(32rem,92vw)] overflow-hidden p-0 max-lg:[&_[data-slot=dialog-close]]:!size-[44px] sm:max-w-lg"
          aria-describedby={undefined}
        >
          <div className="flex max-h-[85vh] flex-col">
            <DialogHeader className="shrink-0 border-b border-border/40 px-4 pb-4 pt-6 sm:px-6">
              <DialogTitle>{msg("share.dialog_title")}</DialogTitle>
            </DialogHeader>

            {state === null ? (
              <div className="flex items-center justify-center gap-2 px-6 py-10 text-sm text-muted-foreground">
                <CircleNotch className="size-4 animate-spin" />
                {msg("share.loading")}
              </div>
            ) : (
              <>
                <div className="shrink-0 border-b border-border/40 px-4 py-4 sm:px-6">
                  <InvitePeople
                    ownerName={state.owner}
                    onInvite={handleInvite}
                    canTransfer={isOwner}
                    onTransfer={setTransferTarget}
                  />
                </div>

                <div className="shrink-0 px-4 pb-1 pt-3 sm:px-6">
                  <p className="text-sm font-medium text-foreground">
                    {msg("share.people_with_access")}
                    <span className="ms-1.5 text-xs font-normal tabular-nums text-muted-foreground">
                      {accessCount}
                    </span>
                  </p>
                </div>

                <div className="min-h-0 flex-1 overflow-y-auto px-4 sm:px-6">
                  {state.owner && (
                    <SettingsRow
                      icon={User}
                      label={
                        <span
                          dir="ltr"
                          title={state.owner}
                          className="inline-block max-w-[200px] truncate align-bottom font-mono"
                        >
                          {state.owner}
                        </span>
                      }
                      description={state.owner.toLowerCase() === me ? msg("share.you") : undefined}
                    >
                      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        {msg("share.owner_label")}
                      </span>
                    </SettingsRow>
                  )}

                  {state.members.map((member) =>
                    member.username === me ? (
                      <SettingsRow
                        key={member.username}
                        icon={User}
                        label={
                          <span
                            dir="ltr"
                            title={member.username}
                            className="inline-block max-w-[200px] truncate align-bottom font-semibold text-foreground"
                          >
                            {member.username}
                          </span>
                        }
                        description={msg("share.you")}
                      >
                        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          {roleLabel(member.role)}
                        </span>
                      </SettingsRow>
                    ) : (
                      <SettingsRow
                        key={member.username}
                        icon={User}
                        label={
                          <span
                            dir="ltr"
                            title={member.username}
                            className="inline-block max-w-[200px] truncate align-bottom font-semibold text-foreground"
                          >
                            {member.username}
                          </span>
                        }
                      >
                        <Select
                          value={member.role}
                          onValueChange={(next) => {
                            if (next === TRANSFER_VALUE) {
                              setTransferTarget(member.username);
                            } else {
                              void handleRoleChange(member.username, next as MemberRole);
                            }
                          }}
                        >
                          <SelectTrigger
                            size="sm"
                            className="!h-[44px] min-w-[120px] lg:!h-8"
                            aria-label={msg("share.role.change_aria")}
                          >
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {ROLE_OPTIONS.map((option) => (
                              <SelectItem key={option} value={option}>
                                {roleLabel(option)}
                              </SelectItem>
                            ))}
                            {isOwner && (
                              <SelectItem value={TRANSFER_VALUE}>
                                {msg("share.transfer.action")}
                              </SelectItem>
                            )}
                          </SelectContent>
                        </Select>
                        <TooltipButton tooltip={msg("share.remove_member_aria")}>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="text-muted-foreground hover:text-destructive"
                            onClick={() => handleRemove(member.username)}
                            aria-label={msg("share.remove_member_aria")}
                          >
                            <X className="size-3.5" />
                          </Button>
                        </TooltipButton>
                      </SettingsRow>
                    ),
                  )}
                </div>

                <div className="shrink-0 space-y-3 border-t border-border/40 px-4 py-4 sm:px-6">
                  <SettingsRow
                    icon={state.general_access === "anyone" ? Globe : Lock}
                    label={msg("share.general_access")}
                    description={
                      state.general_access === "restricted"
                        ? msg("share.general_access.restricted_desc")
                        : undefined
                    }
                  >
                    <div className="flex w-[140px] flex-wrap items-center justify-end gap-2 sm:w-auto">
                      <Select
                        value={state.general_access}
                        onValueChange={(next) => handleAccessChange(next as GeneralAccess)}
                        disabled={savingAccess}
                      >
                        <SelectTrigger size="sm" className="!h-[44px] min-w-[140px] lg:!h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="restricted">
                            {msg("share.general_access.restricted")}
                          </SelectItem>
                          <SelectItem value="anyone">
                            {msg("share.general_access.anyone")}
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      {state.general_access === "anyone" && (
                        <Select
                          value={state.general_role}
                          onValueChange={(next) => handleLinkRoleChange(next as LinkRole)}
                          disabled={savingAccess}
                        >
                          <SelectTrigger
                            size="sm"
                            className="!h-[44px] min-w-[104px] lg:!h-8"
                            aria-label={msg("share.role.change_aria")}
                          >
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="viewer">{roleLabel("viewer")}</SelectItem>
                            <SelectItem value="editor">{roleLabel("editor")}</SelectItem>
                          </SelectContent>
                        </Select>
                      )}
                    </div>
                  </SettingsRow>

                  {shareUrl && state.general_access === "anyone" && (
                    <div
                      dir="ltr"
                      className="flex items-center gap-1 rounded-md border border-input bg-background ps-3 pe-1 transition-[color,box-shadow,border-color] duration-120 ease-[cubic-bezier(0.2,0.8,0.2,1)] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50"
                    >
                      <code className="min-w-0 flex-1 truncate py-2 font-mono text-[0.6875rem] text-muted-foreground">
                        {shareUrl}
                      </code>
                      <div aria-hidden className="h-5 w-px shrink-0 bg-border/70" />
                      <TooltipButton tooltip={msg("share.copy_link")}>
                        <CopyButton
                          text={shareUrl}
                          ariaLabel={msg("share.copy_link")}
                          onCopied={() => toast.success(msg("share.link_copied"))}
                          onCopyError={() => toast.error(msg("clipboard.copy_failed"))}
                          className="size-7 shrink-0 text-muted-foreground hover:text-foreground focus-visible:bg-accent focus-visible:text-foreground focus-visible:ring-0"
                        />
                      </TooltipButton>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={transferTarget !== null}
        onOpenChange={(next) => {
          if (!next) setTransferTarget(null);
        }}
      >
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogTitleRow
            title={msg("share.transfer.confirm_title")}
            description={
              <>
                {transferBodyBefore}
                <bdi className="font-mono font-medium text-foreground">{transferTarget}</bdi>
                {transferBodyAfter}
              </>
            }
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setTransferTarget(null)}
              disabled={transferring}
              className="w-full justify-center"
            >
              {msg("share.transfer.cancel")}
            </Button>
            <Button
              onClick={handleTransfer}
              disabled={transferring}
              className="w-full justify-center shadow-xs"
            >
              {transferring ? (
                <CircleNotch className="size-4 animate-spin" />
              ) : (
                msg("share.transfer.confirm_cta")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Username autocomplete + role picker to add a new member grant. */
function InvitePeople({
  ownerName,
  onInvite,
  canTransfer,
  onTransfer,
}: {
  ownerName: string | null;
  onInvite: (username: string, role: MemberRole) => Promise<void>;
  canTransfer: boolean;
  onTransfer: (username: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<string[]>([]);
  const [searching, setSearching] = useState(false);
  const [role, setRole] = useState<MemberRole | typeof TRANSFER_VALUE>("viewer");
  const [inviting, setInviting] = useState(false);
  const [open, setOpen] = useState(false);
  const lastQuery = useRef("");

  const runSearch = useCallback((prefix: string) => {
    lastQuery.current = prefix;
    if (prefix.trim().length === 0) {
      setResults([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    searchUsers(prefix)
      .then((res) => {
        if (lastQuery.current !== prefix) return;
        setResults(res.usernames);
      })
      .catch(() => {
        if (lastQuery.current === prefix) setResults([]);
      })
      .finally(() => {
        if (lastQuery.current === prefix) setSearching(false);
      });
  }, []);

  useEffect(() => {
    const id = setTimeout(() => runSearch(query), 200);
    return () => clearTimeout(id);
  }, [query, runSearch]);

  const submit = async (username: string) => {
    const target = username.trim();
    if (target.length === 0 || inviting) return;
    if (ownerName && target === ownerName) {
      toast.error(msg("share.cannot_grant_self"));
      return;
    }
    if (role === TRANSFER_VALUE) {
      // Hand ownership off straight from the invite row; the parent confirms,
      // then (if needed) adds the user before transferring.
      onTransfer(target);
      setQuery("");
      setResults([]);
      setOpen(false);
      setRole("viewer");
      return;
    }
    setInviting(true);
    try {
      await onInvite(target, role);
      setQuery("");
      setResults([]);
      setOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("share.save_failed"));
    } finally {
      setInviting(false);
    }
  };

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-foreground">{msg("share.invite_label")}</p>
      <div className="relative">
        <div className="flex items-center gap-1 rounded-md border border-input bg-background ps-3 pe-1 transition-[color,box-shadow,border-color] duration-120 ease-[cubic-bezier(0.2,0.8,0.2,1)] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50">
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => window.setTimeout(() => setOpen(false), 150)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submit(query);
              } else if (e.key === "Escape") {
                setOpen(false);
              }
            }}
            placeholder={msg("share.invite_placeholder")}
            aria-label={msg("share.invite_label")}
            disabled={inviting}
            dir="ltr"
            className="!h-[44px] flex-1 rounded-none border-0 bg-transparent px-0 text-xs shadow-none backdrop-blur-none focus-visible:border-transparent focus-visible:ring-0 lg:!h-8"
          />
          <div aria-hidden className="h-5 w-px shrink-0 bg-border/70" />
          <Select
            value={role}
            onValueChange={(next) => setRole(next as MemberRole | typeof TRANSFER_VALUE)}
          >
            <SelectTrigger
              size="sm"
              className="!h-[44px] gap-1 rounded-md border-0 bg-transparent px-2 text-xs shadow-none hover:border-transparent hover:bg-accent/55 hover:shadow-none focus-visible:border-transparent focus-visible:bg-accent/55 focus-visible:ring-0 data-[state=open]:border-transparent data-[state=open]:bg-accent/60 data-[state=open]:shadow-none lg:!h-7"
              aria-label={msg("share.role.change_aria")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ROLE_OPTIONS.map((option) => (
                <SelectItem key={option} value={option}>
                  {roleLabel(option)}
                </SelectItem>
              ))}
              {canTransfer && (
                <SelectItem value={TRANSFER_VALUE}>{msg("share.transfer.action")}</SelectItem>
              )}
            </SelectContent>
          </Select>
          <TooltipButton tooltip={msg("share.invite")}>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => void submit(query)}
              disabled={inviting || query.trim().length === 0}
              aria-label={msg("share.invite")}
              className="!size-[44px] shrink-0 text-muted-foreground hover:text-foreground focus-visible:bg-accent focus-visible:text-foreground focus-visible:ring-0 lg:!size-7"
            >
              {inviting ? (
                <CircleNotch className="size-4 animate-spin" />
              ) : (
                <UserPlus className="size-4" />
              )}
            </Button>
          </TooltipButton>
        </div>
        {open && query.trim().length > 0 && (
          <div className="absolute inset-x-0 top-full z-30 mt-2 max-h-48 overflow-y-auto rounded-md border border-border/70 bg-popover shadow-lg">
            {searching ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
                <CircleNotch className="size-3.5 animate-spin" />
                {msg("share.searching")}
              </div>
            ) : results.length === 0 ? (
              <div className="px-3 py-2 text-xs text-muted-foreground">
                {msg("share.no_results")}
              </div>
            ) : (
              <ul role="listbox" className="py-1">
                {results.map((name) => (
                  <li key={name}>
                    <button
                      type="button"
                      dir="ltr"
                      onMouseDown={(e) => {
                        e.preventDefault();
                        void submit(name);
                      }}
                      className="flex min-h-[44px] w-full items-center px-3 py-1.5 text-start font-mono text-xs hover:bg-accent/60 lg:min-h-0"
                    >
                      {name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
      {role !== TRANSFER_VALUE && roleDesc(role) ? (
        <p className="text-xs text-muted-foreground/80">{roleDesc(role)}</p>
      ) : null}
    </div>
  );
}
