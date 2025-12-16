// import Image from "next/image";

// export default function Home() {
//   return (
//     <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
//       <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
//         <Image
//           className="dark:invert"
//           src="/next.svg"
//           alt="Next.js logo"
//           width={100}
//           height={20}
//           priority
//         />
//         <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
//           <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
//             To get started, edit the page.tsx file.
//           </h1>
//           <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
//             Looking for a starting point or more instructions? Head over to{" "}
//             <a
//               href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Templates
//             </a>{" "}
//             or the{" "}
//             <a
//               href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Learning
//             </a>{" "}
//             center.
//           </p>
//         </div>
//         <div className="flex flex-col gap-4 text-base font-medium sm:flex-row">
//           <a
//             className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-foreground px-5 text-background transition-colors hover:bg-[#383838] dark:hover:bg-[#ccc] md:w-[158px]"
//             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             <Image
//               className="dark:invert"
//               src="/vercel.svg"
//               alt="Vercel logomark"
//               width={16}
//               height={16}
//             />
//             Deploy Now
//           </a>
//           <a
//             className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-black/[.08] px-5 transition-colors hover:border-transparent hover:bg-black/[.04] dark:border-white/[.145] dark:hover:bg-[#1a1a1a] md:w-[158px]"
//             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Documentation
//           </a>
//         </div>
//       </main>
//     </div>
//   );
// }


import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-black px-6">
      <div className="max-w-3xl w-full bg-white dark:bg-zinc-900 rounded-2xl shadow-md p-10 border border-gray-200 dark:border-zinc-800">

        {/* Header Logo */}
        <div className="flex items-center justify-center mb-8">
          <Image
            src="/next.svg"
            alt="App Logo"
            width={120}
            height={40}
            className="opacity-80 dark:invert"
          />
        </div>

        {/* Main Text */}
        <h1 className="text-3xl font-semibold text-center text-gray-800 dark:text-zinc-100 mb-4">
          CAPEX Intelligence Platform
        </h1>

        <p className="text-lg text-center text-gray-600 dark:text-zinc-400 max-w-xl mx-auto mb-10">
          Predict CAPEX, explore feature importance, and chat with an AI assistant —
          all in one seamless dashboard built for Industry 4.0.
        </p>

        {/* Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-4">
          
          <a
            href="/predict"
            className="w-full sm:w-auto px-6 py-3 rounded-full bg-blue-600 text-white font-medium shadow hover:bg-blue-700 transition"
          >
            Start Prediction
          </a>

          <a
            href="/dashboard"
            className="w-full sm:w-auto px-6 py-3 rounded-full border border-gray-300 dark:border-zinc-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-zinc-800 transition"
          >
            View Dashboard
          </a>

          <a
            href="/assistant"
            className="w-full sm:w-auto px-6 py-3 rounded-full border border-gray-300 dark:border-zinc-700 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-zinc-800 transition"
          >
            Ask the AI
          </a>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-gray-400 dark:text-zinc-500 mt-8">
          Built with Next.js, FastAPI & Tailwind — crafted for Manufacturing Analytics.
        </p>

      </div>
    </div>
  );
}
