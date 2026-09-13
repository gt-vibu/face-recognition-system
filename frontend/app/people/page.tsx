"use client"

import * as React from "react"
import Link from "next/link"
import { CircleX, Plus, Trash } from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { PersonAvatar } from "@/components/person-avatar"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button, buttonVariants } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, errorMessage, type Person } from "@/lib/api"
import { cn } from "@/lib/utils"

export default function PeoplePage() {
  const [people, setPeople] = React.useState<Person[] | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const load = React.useCallback(() => {
    api
      .persons()
      .then((ps) => {
        setPeople(ps)
        setError(null)
      })
      .catch((e) => setError(errorMessage(e)))
  }, [])

  React.useEffect(load, [load])

  async function remove(person: Person) {
    try {
      await api.deletePerson(person.person_id)
    } catch (e) {
      setError(errorMessage(e))
    }
    load()
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 sm:pt-2">
      <PageHeader title="Enrolled people" description="Everyone who can be identified by the system." />

      {error && (
        <Alert variant="destructive" className="border-red-200 bg-red-50">
          <CircleX />
          <AlertTitle>Something went wrong</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <section className="rounded-lg border border-border bg-white shadow-[0_1px_2px_rgb(0_0_0/0.03)]">
        <div className="flex items-center justify-between gap-3 px-5 pt-5 pb-4 sm:px-6">
          <div className="flex items-baseline gap-3">
            <h2 className="text-sm font-medium">People</h2>
            <span className="text-sm text-muted-foreground tabular-nums">{people ? people.length : "—"}</span>
          </div>
          <Link href="/enroll" className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
            <Plus /> Enroll person
          </Link>
        </div>

        {people === null && !error && (
          <div className="space-y-2 px-5 pb-5 sm:px-6" aria-busy>
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-12 animate-pulse rounded-md bg-stone-100" />
            ))}
          </div>
        )}

        {people && people.length === 0 && (
          <p className="border-t border-border px-5 py-10 text-center text-sm text-muted-foreground sm:px-6">
            No one is enrolled yet.{" "}
            <Link href="/enroll" className="font-medium text-foreground underline-offset-4 hover:underline">
              Enroll a person
            </Link>
          </p>
        )}

        {people && people.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow className="border-t border-border hover:bg-transparent">
                <TableHead className="h-9 pl-5 text-xs font-normal text-muted-foreground sm:pl-6">Name</TableHead>
                <TableHead className="h-9 text-xs font-normal text-muted-foreground">Samples</TableHead>
                <TableHead className="hidden h-9 text-xs font-normal text-muted-foreground sm:table-cell">
                  Enrolled
                </TableHead>
                <TableHead className="h-9 pr-5 sm:pr-6">
                  <span className="sr-only">Actions</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {people.map((person) => (
                <TableRow key={person.person_id} className="group border-border hover:bg-stone-50/70">
                  <TableCell className="py-2.5 pl-5 sm:pl-6">
                    <div className="flex items-center gap-3">
                      <PersonAvatar person={person} name={person.name} className="size-9 text-xs" />
                      <span className="font-medium">{person.name}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground tabular-nums">{person.num_samples}</TableCell>
                  <TableCell className="hidden text-muted-foreground tabular-nums sm:table-cell">
                    {person.created_at.slice(0, 10)}
                  </TableCell>
                  <TableCell className="w-12 pr-5 text-right sm:pr-6">
                    <AlertDialog>
                      <AlertDialogTrigger
                        render={
                          <Button
                            variant="ghost"
                            size="icon"
                            aria-label={`Delete ${person.name}`}
                            title={`Delete ${person.name}`}
                            className="text-muted-foreground transition-opacity duration-150 hover:bg-red-50 hover:text-red-700 focus-visible:opacity-100 group-focus-within:opacity-100 group-hover:opacity-100 [@media(hover:hover)]:opacity-0"
                          />
                        }
                      >
                        <Trash />
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete {person.name}?</AlertDialogTitle>
                          <AlertDialogDescription>
                            Their stored face embedding will be removed from the local database. They will need to be
                            enrolled again to be recognised.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogCancel variant="destructive" onClick={() => remove(person)}>
                            Delete
                          </AlertDialogCancel>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}

        {people && people.length > 0 && <div className="h-2" aria-hidden />}
      </section>
    </div>
  )
}
