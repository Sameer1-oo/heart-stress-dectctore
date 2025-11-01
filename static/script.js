// Fetch live data from Flask backend every 2 seconds
setInterval(async () => {
  try {
    const res = await fetch("/get_data");
    const data = await res.json();
    document.getElementById("blink").innerText = data.blink_rate;
    const stress = document.getElementById("stress");
    stress.innerText = data.stress;

    // Change stress color dynamically
    if (data.stress === "HIGH") stress.style.color = "red";
    else if (data.stress === "MEDIUM") stress.style.color = "orange";
    else stress.style.color = "lime";
  } catch (err) {
    console.error("Error fetching data:", err);
  }
}, 2000);
