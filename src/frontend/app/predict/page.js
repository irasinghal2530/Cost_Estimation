

// "use client";

// import { useState, useEffect } from "react";
// import axios from "axios";

// const API = "http://localhost:8000";

// const NUMERIC_FIELDS = new Set([
//   "Plant_Age", "Lifetime_Volume", "Target_Annual_Volume",
//   "Variants", "Number_of_Parts", "Avg_Part_Complexity",
//   "BIW_Weight", "Stamping_Dies", "Injection_Molds",
//   "Casting_Tools", "Jigs_and_Fixtures", "Assembly_Line_Equipment",
//   "Robotics", "Paint_Shop_Mods"
// ]);

// export default function PredictPage() {
//   const [form, setForm] = useState({});
//   const [categories, setCategories] = useState({});
//   const [loading, setLoading] = useState(false);
//   const [result, setResult] = useState(null);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     axios.get(`${API}/categories`)
//       .then(res => setCategories(res.data))
//       .catch(() => setError("Failed to load categories"));
//   }, []);

//   const ALL_FIELDS = [
//     "Vehicle_Type", "Material_Type", "Drivetrain", "Automation_Level",
//     "Plant_Age", "Line_Reuse", "Lifetime_Volume", "Target_Annual_Volume",
//     "Variants", "Number_of_Parts", "Avg_Part_Complexity", "BIW_Weight",
//     "Stamping_Dies", "Injection_Molds", "Casting_Tools", "Jigs_and_Fixtures",
//     "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
//   ];

//   const updateField = (field, value) => {
//     setForm(prev => ({
//       ...prev,
//       [field]: NUMERIC_FIELDS.has(field)
//         ? (value === "" ? null : Number(value))
//         : value,
//     }));
//   };


//   const predict = async () => {
//     setLoading(true);
//     setResult(null);
//     setError(null);

//     try {
//       const res = await axios.post(`${API}/predict`, form);
//       setResult(res.data.predicted_CAPEX);
//     } catch (err) {
//       setError(err?.response?.data?.detail || "Prediction failed");
//     } finally {
//       setLoading(false);
//     }
//   };

//   const isDropdown = (field) => {
//     const key = field.toLowerCase();
//     return Object.keys(categories).includes(key);
//   };

//   return (
//     <div className="min-h-screen bg-gray-100 p-8 flex justify-center">
//       <div className="w-full max-w-4xl">

//         {/* Page Header */}
//         <h1 className="text-3xl font-bold text-gray-800 mb-6">
//           CAPEX Prediction
//         </h1>

//         {/* Form Card */}
//         <div className="bg-white shadow-lg rounded-2xl p-8 border border-gray-200">

//           <h2 className="text-xl font-semibold text-gray-700 mb-4">
//             Enter Project Parameters
//           </h2>

//           <div className="grid grid-cols-2 gap-6">
//             {ALL_FIELDS.map((field) => {
//               const key = field.toLowerCase();
//               const dropdownOptions = categories[key];

//               return (
//                 <div key={field} className="flex flex-col">
//                   <label className="text-sm font-medium text-gray-600 mb-1">
//                     {field.replace(/_/g, " ")}
//                   </label>

//                   {isDropdown(field) ? (
//                     <select
//                       onChange={(e) => updateField(field, e.target.value)}
//                       className="p-2 rounded-lg border border-gray-300 bg-gray-50 hover:bg-gray-100 transition"
//                     >
//                       <option value="">Select {field}</option>
//                       {dropdownOptions?.map((opt) => (
//                         <option key={opt} value={opt}>
//                           {opt}
//                         </option>
//                       ))}
//                     </select>
//                   ) : (
//                     <input
//                       type={NUMERIC_FIELDS.has(field) ? "number" : "text"}
//                       placeholder={field}
//                       onChange={(e) => updateField(field, e.target.value)}
//                       className="p-2 rounded-lg border border-gray-300 bg-gray-50 hover:bg-gray-100 transition"
//                     />
//                   )}
//                 </div>
//               );
//             })}
//           </div>

//           {/* Predict Button */}
//           <button
//             onClick={predict}
//             disabled={loading}
//             className="mt-6 w-full py-3 bg-blue-600 text-white font-semibold rounded-lg shadow hover:bg-blue-700 transition disabled:opacity-60"
//           >
//             {loading ? "Predicting..." : "Predict CAPEX"}
//           </button>

//           {/* Error Message */}
//           {error && (
//             <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-lg border border-red-300">
//               {String(error)}
//             </div>
//           )}

//           {/* Result Card */}
//           {result !== null && (
//             <div className="mt-6 p-5 bg-green-50 border border-green-200 rounded-xl text-lg shadow">
//               <span className="font-semibold text-green-700">
//                 Predicted CAPEX:&nbsp;
//               </span>
//               <span className="font-bold text-green-800 text-xl">
//                 {result}
//               </span>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }


"use client";

import { useState, useEffect } from "react";
import axios from "axios";

const API = "http://localhost:8000";

// Numeric fields (everything else becomes dropdown or text)
const NUMERIC_FIELDS = new Set([
  "Plant_Age", "Lifetime_Volume", "Target_Annual_Volume",
  "Variants", "Number_of_Parts", "Avg_Part_Complexity",
  "BIW_Weight", "Stamping_Dies", "Injection_Molds",
  "Casting_Tools", "Jigs_and_Fixtures", "Assembly_Line_Equipment",
  "Robotics", "Paint_Shop_Mods"
]);

// All fields in correct order
const ALL_FIELDS = [
  "Vehicle_Type", "Material_Type", "Drivetrain", "Automation_Level",
  "Plant_Age", "Line_Reuse", "Lifetime_Volume", "Target_Annual_Volume",
  "Variants", "Number_of_Parts", "Avg_Part_Complexity", "BIW_Weight",
  "Stamping_Dies", "Injection_Molds", "Casting_Tools", "Jigs_and_Fixtures",
  "Assembly_Line_Equipment", "Robotics", "Paint_Shop_Mods"
];

export default function PredictPage() {
  // Initialize form with empty values
  const initialForm = ALL_FIELDS.reduce((acc, f) => {
    acc[f] = "";
    return acc;
  }, {});

  const [form, setForm] = useState(initialForm);
  const [categories, setCategories] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Load dropdown categories
  useEffect(() => {
    axios
      .get(`${API}/categories`)
      .then(res => setCategories(res.data))
      .catch(() => setError("Failed to load categories"));
  }, []);

  // Update field in form
  const updateField = (field, value) => {
    setForm(prev => ({
      ...prev,
      [field]: NUMERIC_FIELDS.has(field)
        ? (value === "" ? undefined : Number(value))
        : value,
    }));
  };

  // Determine if a field should be dropdown
  const isDropdown = (field) => {
    const key = field.toLowerCase();
    return categories[key] !== undefined;
  };

  // Predict CAPEX
  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    console.log("Frontend sending:", form); // debug

    try {
      const res = await axios.post(`${API}/predict`, form);
      setResult(res.data.predicted_CAPEX);
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8 flex justify-center">
      <div className="w-full max-w-4xl">
        
        <h1 className="text-3xl font-bold text-gray-800 mb-6">
          CAPEX Prediction
        </h1>

        <div className="bg-white shadow-lg rounded-2xl p-8 border border-gray-200">

          <h2 className="text-xl font-semibold text-gray-700 mb-4">
            Enter Project Parameters
          </h2>

          {/* FORM GRID */}
          <div className="grid grid-cols-2 gap-6">
            {ALL_FIELDS.map((field) => {
              const key = field.toLowerCase();
              const options = categories[key];

              return (
                <div key={field} className="flex flex-col">
                  <label className="text-sm font-medium text-gray-600 mb-1">
                    {field.replace(/_/g, " ")}
                  </label>

                  {/* DROPDOWN FIELDS */}
                  {isDropdown(field) ? (
                    <select
                      value={form[field] || ""}
                      onChange={(e) => updateField(field, e.target.value)}
                      className="p-2 rounded-lg border border-gray-300 bg-gray-50 hover:bg-gray-100 transition"
                    >
                      <option value="">Select {field}</option>
                      {options?.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    // NUMERIC/TEXT FIELDS
                    <input
                      type={NUMERIC_FIELDS.has(field) ? "number" : "text"}
                      value={form[field] || ""}
                      placeholder={field}
                      onChange={(e) => updateField(field, e.target.value)}
                      className="p-2 rounded-lg border border-gray-300 bg-gray-50 hover:bg-gray-100 transition"
                    />
                  )}

                </div>
              );
            })}
          </div>

          {/* PREDICT BUTTON */}
          <button
            onClick={predict}
            disabled={loading}
            className="mt-6 w-full py-3 bg-blue-600 text-white font-semibold rounded-lg shadow hover:bg-blue-700 transition disabled:opacity-60"
          >
            {loading ? "Predicting..." : "Predict CAPEX"}
          </button>

          {/* ERROR */}
          {error && (
            <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-lg border border-red-300">
              {String(error)}
            </div>
          )}

          {/* RESULT */}
          {result !== null && (
            <div className="mt-6 p-5 bg-green-50 border border-green-200 rounded-xl text-lg shadow">
              <span className="font-semibold text-green-700">
                Predicted CAPEX:{" "}
              </span>
              <span className="font-bold text-green-800 text-xl">
                {result}
              </span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

