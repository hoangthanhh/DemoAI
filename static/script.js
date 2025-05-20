let previousWarning = "";

function checkWarning() {
    fetch('/get_warning')
        .then(response => response.text())
        .then(warning => {
            const warningList = document.getElementById("warning-list");

            if (warning && warning.length > 0 && warning !== previousWarning) {
                warningList.innerHTML = "";
                const warningItems = warning.split("!").filter(w => w.trim() !== "");
                warningItems.forEach(w => {
                    const li = document.createElement("li");
                    li.innerText = w.trim() + "!";
                    warningList.appendChild(li);
                });
                document.getElementById("alert-audio").play();
                previousWarning = warning;
            }

            if (!warning || warning.trim() === "") {
                warningList.innerHTML = "<li style='color: #4caf50;'>Không có cảnh báo</li>";
                previousWarning = "";
            }
        });
}

setInterval(checkWarning, 1000);
