function populateForm(boardData) {
    document.getElementById('charge_count').value = boardData.max_charges;
    document.getElementById('current_turn').value = boardData.current_turn;

    // Populate charges, shotgun, items, and active items for each player
    boardData.charges.forEach((charge, index) => {
        document.getElementById(`p${index+1}_charges`).value = charge;
    });
    boardData.items.forEach((playerItems, index) => {
        Object.entries(playerItems).forEach(([item, count]) => {
            document.getElementById(`p${index+1}_${item}`).value = count;
        });
    });
    Object.entries(boardData.active_items).forEach(([item, count]) => {
        document.getElementById(`active_${item}`).value = count;
    });

    // Set the checkbox values
    document.getElementById('skip_next').checked = boardData.skip_next;
    document.getElementById('chamber_public').checked = boardData.chamber_public;

    // Note: This code assumes that your form fields' ids are named consistently with the boardData keys.
    // Adjust the ids as necessary to match your form.
}

document.getElementById('board_mod').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from submitting via the browser.

    const formData = new FormData(this);
    fetch('/modify_board', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())  // Convert response to text (or .json() if JSON response)
    .then(data => {
        document.getElementById('outputs').innerText = data; // Display the response
    })
    .catch(error => console.error('Error:', error));
});