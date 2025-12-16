// "use client";

// import { useEffect, useState } from "react";
// import axios from "axios";
// import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// export default function Dashboard() {
//   const [featureImp, setFeatureImp] = useState([]);

//   useEffect(() => {
//     axios
//       .get("http://localhost:8000/feature_importance?n=10")
//       .then((res) => setFeatureImp(res.data.top_features))
//       .catch(console.error);
//   }, []);

//   return (
//     <div>
//       <h1 className="text-2xl font-bold mb-4">Feature Importance</h1>

//       <div className="w-full h-[400px] bg-white p-4 rounded shadow">
//         <ResponsiveContainer>
//           <BarChart data={featureImp}>
//             <XAxis dataKey="feature" />
//             <YAxis />
//             <Tooltip />
//             <Bar dataKey="importance" />
//           </BarChart>
//         </ResponsiveContainer>
//       </div>
//     </div>
//   );
// }


"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function Dashboard() {
  const [featureImp, setFeatureImp] = useState([]);

  useEffect(() => {
    axios
      .get("http://localhost:8000/feature_importance?n=10")
      .then((res) => setFeatureImp(res.data.top_features))
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen w-full bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">

        <h1 className="text-3xl font-extrabold text-gray-800 mb-6">
          Feature Importance
        </h1>

        <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
          <h2 className="text-lg font-semibold mb-4 text-gray-700">
            Top Features (Model Explanation)
          </h2>

          <div className="w-full h-[420px]">
            <ResponsiveContainer>
              <BarChart data={featureImp}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
                <XAxis dataKey="feature" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="importance" className="fill-blue-500" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
