export default function HistoryTable({ items, onSelect, loading }) {
  if (loading) return <p className="muted">Loading history…</p>;
  if (!items || items.length === 0) {
    return <p className="muted">No predictions yet — try one above.</p>;
  }

  return (
    <table className="history-table">
      <thead>
        <tr>
          <th>SMILES</th>
          <th>Predicted log(S)</th>
          <th>Model</th>
          <th>When</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id} onClick={() => onSelect(item.id)} className="clickable-row">
            <td>
              <code>{item.original_smiles}</code>
            </td>
            <td>{item.predicted_log_solubility.toFixed(3)}</td>
            <td>{item.model_name}</td>
            <td>{new Date(item.created_at).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
