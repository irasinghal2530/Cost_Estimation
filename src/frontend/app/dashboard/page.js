
// "use client";

// import { useEffect, useState } from "react";
// import axios from "axios";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   CartesianGrid,
//   LineChart,
//   Line,
// } from "recharts";

// export default function Dashboard() {
//   const [featureImp, setFeatureImp] = useState([]);
//   const [metrics, setMetrics] = useState(null);
//   const [recentPreds, setRecentPreds] = useState([]);


// useEffect(() => {
//   const API_BASE = "http://127.0.0.1:8000";

//   axios.get(`${API_BASE}/feature_importance?n=10`)
//     .then(res => setFeatureImp(res.data.top_features))
//     .catch(console.error);

//   axios.get(`${API_BASE}/metrics`)
//     .then(res => setMetrics(res.data))
//     .catch(console.error);

//   axios.get(`${API_BASE}/recent_predictions`)
//     .then(res => setRecentPreds(res.data))
//     .catch(console.error);
// }, []);


//   return (
//     <div className="min-h-screen p-6 space-y-10 bg-gray-100">
      
//       {/* KPI METRICS */}
//       {metrics && (
//         <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
//           <Metric title="R² Score" value={metrics.r2.toFixed(4)} />
//           <Metric title="RMSE" value={`$${metrics.rmse}M`} />
//           <Metric title="MAE" value={`$${metrics.mae}M`} />
//           <Metric title="MAPE" value={`${metrics.mape}%`} />
//         </div>
//       )}

//       {/* FEATURE IMPORTANCE */}
//       <Section title="Top Feature Importance">
//         <ChartContainer height={420}>
//           <ResponsiveContainer>
//             <BarChart data={featureImp} layout="vertical">
//               <CartesianGrid strokeDasharray="3 3" />
//               <XAxis type="number" />
//               <YAxis dataKey="feature" type="category" width={160} />
//               <Tooltip />
//               <Bar dataKey="importance" />
//             </BarChart>
//           </ResponsiveContainer>
//         </ChartContainer>
//       </Section>

//       {/* RECENT PREDICTIONS */}
//       <Section title="Recent Predictions">
//         <ChartContainer height={300}>
//           <ResponsiveContainer>
//             <LineChart data={recentPreds}>
//               <CartesianGrid strokeDasharray="3 3" />
//               <XAxis dataKey="timestamp" />
//               <YAxis />
//               <Tooltip />
//               <Line dataKey="predicted_capex" strokeWidth={2} />
//             </LineChart>
//           </ResponsiveContainer>
//         </ChartContainer>
//       </Section>

//     </div>
//   );
// }

// /* ---------------- Reusable pieces ---------------- */

// function Metric({ title, value }) {
//   return (
//     <div className="bg-white rounded-xl p-5 shadow">
//       <div className="text-sm text-gray-500">{title}</div>
//       <div className="text-3xl font-bold text-gray-800 mt-2">{value}</div>
//     </div>
//   );
// }

// function Section({ title, children }) {
//   return (
//     <div className="bg-white rounded-xl p-6 shadow">
//       <h2 className="text-lg font-semibold text-gray-700 mb-4">{title}</h2>
//       {children}
//     </div>
//   );
// }

// function ChartContainer({ height, children }) {
//   return <div style={{ height }}>{children}</div>;
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
  LineChart,
  Line,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

export default function Dashboard() {
  const [featureImp, setFeatureImp] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [recentPreds, setRecentPreds] = useState([]);

  useEffect(() => {
    axios
      .get(`${API_BASE}/feature_importance?n=10`)
      .then((res) => setFeatureImp(res.data.top_features ?? []))
      .catch(console.error);

    axios
      .get(`${API_BASE}/metrics`)
      .then((res) => setMetrics(res.data))
      .catch(console.error);

    axios
      .get(`${API_BASE}/recent_predictions`)
      .then((res) => setRecentPreds(res.data ?? []))
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen p-6 space-y-10 bg-gray-100">

      {/* ================= KPIs ================= */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Metric title="R² Score" value={metrics.r2?.toFixed(4)} />
          <Metric title="Adj R²" value={metrics.adj_r2?.toFixed(4)} />
          <Metric title="RMSE" value={`$${metrics.rmse?.toFixed(2)}M`} />
          <Metric title="MAE" value={`$${metrics.mae?.toFixed(2)}M`} />
        </div>
      )}

      {/* ============ Feature Importance ============ */}
      <Section title="Top Feature Importance">
        <ChartContainer height={420}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featureImp} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={180} />
              <Tooltip />
              <Bar dataKey="importance" />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </Section>

      {/* ============ Recent Predictions ============ */}
      <Section title="Recent Predictions">
        <ChartContainer height={300}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={recentPreds}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="predicted_capex"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </Section>

    </div>
  );
}

/* ================= Reusable UI ================= */

function Metric({ title, value }) {
  return (
    <div className="bg-white rounded-xl p-5 shadow">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="text-3xl font-bold text-gray-800 mt-2">
        {value ?? "—"}
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="bg-white rounded-xl p-6 shadow">
      <h2 className="text-lg font-semibold text-gray-700 mb-4">{title}</h2>
      {children}
    </div>
  );
}

function ChartContainer({ height, children }) {
  return <div style={{ height }}>{children}</div>;
}
