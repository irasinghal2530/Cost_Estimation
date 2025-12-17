// "use client";

// import { useState, useEffect } from "react";
// import axios from "axios";

// const API = "http://localhost:8000";

// /* -------------------- CONFIG -------------------- */

// const NUMERIC_FIELDS = new Set([
//   "Plant_Age", "Lifetime_Volume", "Target_Annual_Volume",
//   "Variants", "Number_of_Parts", "Avg_Part_Complexity",
//   "BIW_Weight", "Stamping_Dies", "Injection_Molds",
//   "Casting_Tools", "Jigs_and_Fixtures",
//   "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
// ]);

// const ALL_FIELDS = [
//   "Vehicle_Type", "Material_Type", "Drivetrain", "Automation_Level",
//   "Plant_Age", "Line_Reuse", "Lifetime_Volume", "Target_Annual_Volume",
//   "Variants", "Number_of_Parts", "Avg_Part_Complexity", "BIW_Weight",
//   "Stamping_Dies", "Injection_Molds", "Casting_Tools", "Jigs_and_Fixtures",
//   "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
// ];

// /* -------------------- PAGE -------------------- */

// export default function PredictPage() {
//   const [mode, setMode] = useState("single"); // single | batch

//   /* ---------- Single prediction state ---------- */
//   const initialForm = ALL_FIELDS.reduce((acc, f) => {
//     acc[f] = "";
//     return acc;
//   }, {});

//   const [form, setForm] = useState(initialForm);
//   const [categories, setCategories] = useState({});
//   const [result, setResult] = useState(null);

//   /* ---------- Batch prediction state ---------- */
//   const [file, setFile] = useState(null);
//   const [batchResult, setBatchResult] = useState(null);

//   /* ---------- Shared state ---------- */
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   /* ---------- Load dropdown categories ---------- */
//   useEffect(() => {
//     axios
//       .get(`${API}/categories`)
//       .then(res => setCategories(res.data))
//       .catch(() => setError("Failed to load categories"));
//   }, []);

//   /* ---------- Helpers ---------- */
//   const updateField = (field, value) => {
//     setForm(prev => ({
//       ...prev,
//       [field]: NUMERIC_FIELDS.has(field)
//         ? value === "" ? undefined : Number(value)
//         : value,
//     }));
//   };

//   const isDropdown = (field) =>
//     categories[field.toLowerCase()] !== undefined;

//   /* ---------- Single predict ---------- */
//   const predict = async () => {
//     setLoading(true);
//     setError(null);
//     setResult(null);

//     try {
//       const res = await axios.post(`${API}/predict`, form);
//       setResult(res.data.predicted_CAPEX);
//     } catch (err) {
//       setError(err?.response?.data?.detail || "Prediction failed");
//     } finally {
//       setLoading(false);
//     }
//   };

//   /* ---------- Batch predict ---------- */
//   const batchPredict = async () => {
//     if (!file) return;

//     setLoading(true);
//     setError(null);
//     setBatchResult(null);

//     const formData = new FormData();
//     formData.append("file", file);

//     try {
//       const res = await axios.post(
//         `${API}/batch_predict`, // change ONLY if backend route differs
//         formData,
//         { headers: { "Content-Type": "multipart/form-data" } }
//       );

//       setBatchResult(res.data);
//     } catch (err) {
//       setError(err?.response?.data?.detail || "Batch prediction failed");
//     } finally {
//       setLoading(false);
//     }
//   };

//   /* -------------------- RENDER -------------------- */

//   return (
//     <div className="min-h-screen bg-gray-100 p-8 flex justify-center">
//       <div className="w-full max-w-5xl">

//         {/* ---------- HEADER ---------- */}
//         <h1 className="text-3xl font-bold text-gray-800 mb-6">
//           CAPEX Prediction
//         </h1>

//         {/* ---------- MODE SWITCH ---------- */}
//         <div className="flex gap-4 mb-6">
//           <button
//             onClick={() => setMode("single")}
//             className={`px-4 py-2 rounded-lg ${
//               mode === "single" ? "bg-blue-600 text-white" : "bg-gray-200"
//             }`}
//           >
//             Single Prediction
//           </button>

//           <button
//             onClick={() => setMode("batch")}
//             className={`px-4 py-2 rounded-lg ${
//               mode === "batch" ? "bg-blue-600 text-white" : "bg-gray-200"
//             }`}
//           >
//             Batch Prediction
//           </button>
//         </div>

//         {/* ---------- SINGLE PREDICTION ---------- */}
//         {mode === "single" && (
//           <div className="bg-white shadow-lg rounded-2xl p-8 border">

//             <h2 className="text-xl font-semibold mb-4">
//               Enter Project Parameters
//             </h2>

//             <div className="grid grid-cols-2 gap-6">
//               {ALL_FIELDS.map(field => {
//                 const options = categories[field.toLowerCase()];

//                 return (
//                   <div key={field} className="flex flex-col">
//                     <label className="text-sm mb-1">
//                       {field.replace(/_/g, " ")}
//                     </label>

//                     {isDropdown(field) ? (
//                       <select
//                         value={form[field] || ""}
//                         onChange={e => updateField(field, e.target.value)}
//                         className="p-2 border rounded-lg"
//                       >
//                         <option value="">Select {field}</option>
//                         {options.map(opt => (
//                           <option key={opt} value={opt}>{opt}</option>
//                         ))}
//                       </select>
//                     ) : (
//                       <input
//                         type={NUMERIC_FIELDS.has(field) ? "number" : "text"}
//                         value={form[field] || ""}
//                         onChange={e => updateField(field, e.target.value)}
//                         className="p-2 border rounded-lg"
//                       />
//                     )}
//                   </div>
//                 );
//               })}
//             </div>

//             <button
//               onClick={predict}
//               disabled={loading}
//               className="mt-6 w-full py-3 bg-blue-600 text-white rounded-lg"
//             >
//               {loading ? "Predicting..." : "Predict CAPEX"}
//             </button>

//             {result !== null && (
//               <div className="mt-6 bg-green-50 p-4 rounded-lg">
//                 <strong>Predicted CAPEX:</strong> {result}
//               </div>
//             )}
//           </div>
//         )}

//         {/* ---------- BATCH PREDICTION ---------- */}
//         {mode === "batch" && (
//           <div className="bg-white shadow-lg rounded-2xl p-8 border">

//             <h2 className="text-xl font-semibold mb-4">
//               Upload CSV / Excel
//             </h2>

//             <input
//               type="file"
//               accept=".csv,.xlsx"
//               onChange={e => setFile(e.target.files[0])}
//               className="mb-4"
//             />

//             <button
//               onClick={batchPredict}
//               disabled={!file || loading}
//               className="w-full py-3 bg-blue-600 text-white rounded-lg"
//             >
//               {loading ? "Processing..." : "Run Batch Prediction"}
//             </button>

//             {batchResult && (
//               <div className="mt-6 bg-gray-50 p-4 rounded-lg">
//                 <p><strong>Rows processed:</strong> {batchResult.count}</p>
//               </div>
//             )}
//           </div>
//         )}

//         {/* ---------- ERROR ---------- */}
//         {error && (
//           <div className="mt-6 bg-red-100 text-red-700 p-4 rounded-lg">
//             {error}
//           </div>
//         )}

//       </div>
//     </div>
//   );
// }


"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import Papa from "papaparse";

const API = "http://localhost:8000";

/* -------------------- CONFIG -------------------- */

const NUMERIC_FIELDS = new Set([
  "Plant_Age", "Lifetime_Volume", "Target_Annual_Volume",
  "Variants", "Number_of_Parts", "Avg_Part_Complexity",
  "BIW_Weight", "Stamping_Dies", "Injection_Molds",
  "Casting_Tools", "Jigs_and_Fixtures",
  "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
]);

const ALL_FIELDS = [
  "Vehicle_Type", "Material_Type", "Drivetrain", "Automation_Level",
  "Plant_Age", "Line_Reuse", "Lifetime_Volume", "Target_Annual_Volume",
  "Variants", "Number_of_Parts", "Avg_Part_Complexity", "BIW_Weight",
  "Stamping_Dies", "Injection_Molds", "Casting_Tools", "Jigs_and_Fixtures",
  "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
];

/* -------------------- PAGE -------------------- */

export default function PredictPage() {
  const [mode, setMode] = useState("single"); // single | batch

  /* ---------- Single prediction state ---------- */
  const initialForm = ALL_FIELDS.reduce((acc, f) => {
    acc[f] = "";
    return acc;
  }, {});

  const [form, setForm] = useState(initialForm);
  const [categories, setCategories] = useState({});
  const [result, setResult] = useState(null);

  /* ---------- Batch prediction state ---------- */
  const [file, setFile] = useState(null);
  const [batchResult, setBatchResult] = useState(null);

  /* ---------- Shared state ---------- */
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /* ---------- Load dropdown categories ---------- */
  useEffect(() => {
    axios
      .get(`${API}/categories`)
      .then(res => setCategories(res.data))
      .catch(() => setError("Failed to load categories"));
  }, []);

  /* ---------- Helpers ---------- */
  const updateField = (field, value) => {
    setForm(prev => ({
      ...prev,
      [field]: NUMERIC_FIELDS.has(field)
        ? value === "" ? undefined : Number(value)
        : value,
    }));
  };

  const isDropdown = (field) =>
    categories[field.toLowerCase()] !== undefined;

  /* ---------- Single predict ---------- */
  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await axios.post(`${API}/predict`, form);
      setResult(res.data.predicted_CAPEX);
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  /* ---------- Batch predict ---------- */
  const batchPredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setBatchResult(null);

    try {
      let rows = [];

      if (file.name.endsWith(".csv")) {
        // Parse CSV Only
        const csvData = await new Promise((resolve, reject) => {
          Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: results => resolve(results.data),
            error: err => reject(err)
          });
        });
        rows = csvData;
      } else {
        throw new Error("Unsupported file type. Please upload a CSV.");
      }

      // Convert numeric fields
      const cleanedRows = rows.map(row => {
        const cleaned = { ...row };
        Object.keys(cleaned).forEach(k => {
          if (NUMERIC_FIELDS.has(k)) {
            cleaned[k] = cleaned[k] === "" ? undefined : Number(cleaned[k]);
          }
        });
        return cleaned;
      });

      // Send to backend
      const res = await axios.post(`${API}/predict_batch`, { data: cleanedRows });
      setBatchResult(res.data.predictions);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Batch prediction failed");
    } finally {
      setLoading(false);
    }
  };

  /* -------------------- RENDER -------------------- */

  return (
    <div className="min-h-screen bg-gray-100 p-8 flex justify-center">
      <div className="w-full max-w-5xl">

        {/* ---------- HEADER ---------- */}
        <h1 className="text-3xl font-bold text-gray-800 mb-6">
          CAPEX Prediction
        </h1>

        {/* ---------- MODE SWITCH ---------- */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setMode("single")}
            className={`px-4 py-2 rounded-lg ${
              mode === "single" ? "bg-blue-600 text-white" : "bg-gray-200"
            }`}
          >
            Single Prediction
          </button>

          <button
            onClick={() => setMode("batch")}
            className={`px-4 py-2 rounded-lg ${
              mode === "batch" ? "bg-blue-600 text-white" : "bg-gray-200"
            }`}
          >
            Batch Prediction
          </button>
        </div>

        {/* ---------- SINGLE PREDICTION ---------- */}
        {mode === "single" && (
          <div className="bg-white shadow-lg rounded-2xl p-8 border">
            <h2 className="text-xl font-semibold mb-4">
              Enter Project Parameters
            </h2>

            <div className="grid grid-cols-2 gap-6">
              {ALL_FIELDS.map(field => {
                const options = categories[field.toLowerCase()];
                return (
                  <div key={field} className="flex flex-col">
                    <label className="text-sm mb-1">
                      {field.replace(/_/g, " ")}
                    </label>

                    {isDropdown(field) ? (
                      <select
                        value={form[field] || ""}
                        onChange={e => updateField(field, e.target.value)}
                        className="p-2 border rounded-lg"
                      >
                        <option value="">Select {field}</option>
                        {options.map(opt => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={NUMERIC_FIELDS.has(field) ? "number" : "text"}
                        value={form[field] || ""}
                        onChange={e => updateField(field, e.target.value)}
                        className="p-2 border rounded-lg"
                      />
                    )}
                  </div>
                );
              })}
            </div>

            <button
              onClick={predict}
              disabled={loading}
              className="mt-6 w-full py-3 bg-blue-600 text-white rounded-lg"
            >
              {loading ? "Predicting..." : "Predict CAPEX"}
            </button>

            {result !== null && (
              <div className="mt-6 bg-green-50 p-4 rounded-lg">
                <strong>Predicted CAPEX:</strong> {result}
              </div>
            )}
          </div>
        )}

        {/* ---------- BATCH PREDICTION ---------- */}
        {mode === "batch" && (
          <div className="bg-white shadow-lg rounded-2xl p-8 border">
            <h2 className="text-xl font-semibold mb-4">
              Upload CSV
            </h2>

            <input
              type="file"
              accept=".csv"
              onChange={e => setFile(e.target.files[0])}
              className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />

            <button
              onClick={batchPredict}
              disabled={!file || loading}
              className="w-full py-3 bg-blue-600 text-white rounded-lg disabled:bg-gray-400"
            >
              {loading ? "Processing..." : "Run Batch Prediction"}
            </button>

            {batchResult && (
              <div className="mt-6 bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Batch Predictions:</h3>
                <ul className="space-y-1">
                  {batchResult.map((p, i) => (
                    <li key={i} className="bg-green-50 border border-green-200 rounded-lg px-4 py-2">
                      Row {i + 1}: <strong>{p}</strong>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* ---------- ERROR ---------- */}
        {error && (
          <div className="mt-6 bg-red-100 text-red-700 p-4 rounded-lg">
            {error}
          </div>
        )}

      </div>
    </div>
  );
}