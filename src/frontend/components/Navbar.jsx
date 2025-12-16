// "use client";
// import Link from "next/link";

// export default function Navbar() {
//   return (
//     <div className="w-full flex gap-6 p-4 shadow bg-white sticky top-0 z-20">
//       <Link href="/dashboard" className="font-semibold hover:text-blue-600">
//         Dashboard
//       </Link>
//       <Link href="/predict" className="font-semibold hover:text-blue-600">
//         Prediction
//       </Link>
//       <Link href="/assistant" className="font-semibold hover:text-blue-600">
//         AI Assistant
//       </Link>
//     </div>
//   );
// }
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/predict", label: "Prediction" },
    { href: "/assistant", label: "AI Assistant" },
  ];

  return (
    <nav className="
      w-full sticky top-0 z-30
      backdrop-blur-md bg-white/70
      border-b border-slate-200
      shadow-sm
    ">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-8">
        {navItems.map((item) => {
          const active = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                font-medium transition
                ${active 
                  ? "text-blue-600 underline underline-offset-4" 
                  : "text-slate-700 hover:text-blue-600"
                }
              `}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}


